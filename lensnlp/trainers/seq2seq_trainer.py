from pathlib import Path
from typing import List, Union

import random
import logging
from torch.optim.adam import Adam
import torch
import lensnlp.models.nn as nn
from lensnlp.utils.data import Sentence, Corpus, Seq2seqCorpus
from lensnlp.models import RelationExtraction
from lensnlp.utils.training_utils import Metric, clear_embeddings, log_line, init_log, WeightExtractor

from lensnlp.hyper_parameters import Parameter

from torch.optim import Optimizer


log = logging.getLogger('lensnlp')


class Seq2SeqTrainer:

    def __init__(self,
                 model: nn.Model,
                 corpus: Seq2seqCorpus,
                 optimizer: Optimizer = Adam,
                 epoch: int = 0,
                 loss: float = 10000.0,
                 optimizer_state: dict = None,
                 ):

        self.model: nn.Model = model
        self.corpus: Seq2seqCorpus = corpus
        self.optimizer: Optimizer = optimizer
        self.epoch: int = epoch
        self.loss: float = loss
        self.optimizer_state: dict = optimizer_state

    def train(self,
              base_path: Union[Path, str],
              learning_rate: float = 1.0,
              mini_batch_size: int = 20,
              eval_mini_batch_size: int = 20,
              max_epochs: int = 150,
              anneal_factor: float = 0.9,
              train_with_dev: bool = False,
              embeddings_in_memory: bool = True,
              test_mode: bool = False,
              **kwargs
              ) -> dict:

        if eval_mini_batch_size is None:
            eval_mini_batch_size = mini_batch_size

        if type(base_path) is str:
            base_path = Path(base_path)

        init_log(log, base_path)

        log_line(log)
        log.info(f'Evaluation method: Mircro F1-Score')

        weight_extractor = WeightExtractor(base_path)

        optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, **kwargs)

        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        train_data = self.corpus.train

        if train_with_dev:
            train_data.extend(self.corpus.dev)

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []

        previous_metric = 0

        try:

            for epoch in range(0 + self.epoch, max_epochs + self.epoch):
                log_line(log)

                if learning_rate < 0.0001:
                    log_line(log)
                    log.info('learning rate too small - quitting training!')
                    log_line(log)
                    break

                if not test_mode:
                    random.shuffle(train_data)

                batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)]

                self.model.train()

                train_loss: float = 0
                seen_sentences = 0
                modulo = max(1, int(len(batches) / 10))

                for batch_no, batch in enumerate(batches):
                    loss = self.model.forward_loss(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    seen_sentences += len(batch)
                    train_loss += loss.item()

                    clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                    if batch_no % modulo == 0:
                        log.info(f'epoch {epoch + 1} - iter {batch_no}/{len(batches)} - loss '
                                 f'{train_loss / seen_sentences:.8f}')
                        iteration = epoch * len(batches) + batch_no
                        weight_extractor.extract_weights(self.model.state_dict(), iteration)

                train_loss /= len(train_data)

                self.model.eval()

                log_line(log)
                log.info(f'EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.4f}')

                test_metric, test_loss = self._calculate_evaluation_results_for(
                    'TEST', self.corpus.test, embeddings_in_memory, eval_mini_batch_size,
                    base_path / 'test.tsv')
                current_metric = test_metric.micro_avg_f_score()
                if current_metric > previous_metric:
                    self.model.save(base_path / 'best-model.pt')
                    previous_metric = current_metric

                learning_rate /= anneal_factor
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate

            self.model.save(base_path / 'final-model.pt')

        except KeyboardInterrupt:
            log_line(log)
            log.info('Exiting from training early.')
            log.info('Saving model ...')
            self.model.save(base_path / 'final-model.pt')
            log.info('Done.')

        return {'dev_score_history': dev_score_history,
                'train_loss_history': train_loss_history,
                'dev_loss_history': dev_loss_history}

    def _calculate_evaluation_results_for(self,
                                          dataset_name: str,
                                          dataset: List[Sentence],
                                          embeddings_in_memory: bool,
                                          eval_mini_batch_size: int,
                                          out_path: Path = None):

        metric, loss = ReTrainer.evaluate(self.model, dataset, eval_mini_batch_size=eval_mini_batch_size,
                                             embeddings_in_memory=embeddings_in_memory, out_path=out_path)

        log.info(f'{dataset_name:<5}: loss {loss:.8f} - f-score {metric.micro_avg_f_score():.4f} '
                 f'- acc {metric.macro_avg_f_score():.4f}')

        return metric, loss

    @staticmethod
    def evaluate(model: nn.Model, data_set: List[Sentence],
                 eval_mini_batch_size: int = 32,
                 embeddings_in_memory: bool = True,
                 out_path: Path = None) -> (
            dict, float):
        if isinstance(model, RelationExtraction):
            return ReTrainer._evaluate_re_classifier(model, data_set,
                                                     eval_mini_batch_size, embeddings_in_memory, out_path)

    @staticmethod
    def _evaluate_re_classifier(model: nn.Model,
                                  sentences: List[Sentence],
                                  eval_mini_batch_size: int = 20,
                                  embeddings_in_memory: bool = False,
                                  out_path: Path = None) -> (dict, float):

        with torch.no_grad():
            metric = Metric('Evaluation')

            eval_loss = 0

            batches = [sentences[x:x + eval_mini_batch_size] for x in
                       range(0, len(sentences), eval_mini_batch_size)]

            lines: List[str] = []
            for batch in batches:

                labels, loss = model.forward_labels_and_loss(batch)

                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                eval_loss += loss

                sentences_for_batch = [sent.to_plain_string() for sent in batch]
                confidences_for_batch = [[label.score for label in sent_labels] for sent_labels in labels]
                predictions_for_batch = [[label.value for label in sent_labels] for sent_labels in labels]
                true_values_for_batch = [sentence.get_label_names() for sentence in batch]
                available_labels = model.label_dictionary.get_items()

                for sentence, confidence, prediction, true_value in zip(sentences_for_batch, confidences_for_batch,
                                                                        predictions_for_batch, true_values_for_batch):
                    eval_line = '{}\t{}\t{}\t{}\n'.format(sentence, true_value[0], prediction[0], confidence[0])
                    lines.append(eval_line)

                for predictions_for_sentence, true_values_for_sentence in \
                        zip(predictions_for_batch, true_values_for_batch):

                    ReTrainer._evaluate_sentence_for_re_classification(metric,
                                                                       available_labels,
                                                                       predictions_for_sentence,
                                                                       true_values_for_sentence)

            eval_loss /= len(sentences)

            if out_path is not None:
                with open(out_path, "w", encoding='utf-8') as outfile:
                    outfile.write(''.join(lines))

            return metric, eval_loss

    @staticmethod
    def _evaluate_sentence_for_re_classification(metric,
                                                 available_labels: List[str],
                                                 predictions: List[str],
                                                 true_values: List[str]):

        for label in available_labels:
            if label in predictions and label in true_values:
                metric.add_tp(label)
            elif label in predictions and label not in true_values:
                metric.add_fp(label)
            elif label not in predictions and label in true_values:
                metric.add_fn(label)
            elif label not in predictions and label not in true_values:
                metric.add_tn(label)

