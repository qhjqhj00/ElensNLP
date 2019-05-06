from pathlib import Path
from typing import List, Union

import datetime
import random
import logging
from torch.optim.sgd import SGD
import torch
import lensnlp.models.nn as nn
from lensnlp.utilis.data import Sentence, Token, MultiCorpus, Corpus
from lensnlp.models import TextClassifier, SequenceTagger
from lensnlp.utilis.training_utils import Metric, init_output_file, clear_embeddings, EvaluationMetric, \
    log_line, init_log, WeightExtractor

from lensnlp.hyper_parameters import Parameter

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.optim import Optimizer


log = logging.getLogger('lensnlp')


class ModelTrainer:
    """训练器类

            Parameters
            ----------
            model : str
                序列标注模型或者分类模型
            corpus : Corpus
                语料数据
            optimizer : Optimizer
                优化器，默认为SGD
            epoch : int
                epoch数量
            Examples
            --------
             >>> from lensnlp.models import SequenceTagger
             >>> from lensnlp.Embeddings import WordEmbeddings
             >>> from lensnlp.utilis.data_load import load_column_corpus
             >>> corpus = load_column_corpus('./dataset/',{1:'token',2:'ner'},'train.txt','test.txt',lang='UY')
             >>> emb = WordEmbeddings('cn_glove')
             >>> tagger = SequenceTagger(hidden_size=256,embeddings = emb)
             >>> from lensnlp.trainers import ModelTrainer
             >>> trainer = ModelTrainer(tagger,corpus)
             >>> trainer.train()

            """

    def __init__(self,
                 model: nn.Model,
                 corpus: Corpus,
                 optimizer: Optimizer = SGD,
                 epoch:int = 0,
                 loss: float = 10000.0,
                 optimizer_state: dict = None,
                 scheduler_state: dict = None
                 ):

        self.model: nn.Model = model
        self.corpus: Corpus = corpus
        self.optimizer: Optimizer = optimizer
        self.epoch: int = epoch
        self.loss: float = loss
        self.scheduler_state: dict = scheduler_state
        self.optimizer_state: dict = optimizer_state

    def train(self,
              base_path: Union[Path, str],
              evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
              learning_rate: float = Parameter['lr'],
              mini_batch_size: int = Parameter['batch_size'],
              eval_mini_batch_size: int = Parameter['batch_size'],
              max_epochs: int = Parameter['max_epoch'],
              anneal_factor: float = Parameter['anneal_rate'],
              patience: int = Parameter['patience'],
              anneal_against_train_loss: bool = Parameter['anneal_against_train'],
              train_with_dev: bool = Parameter['train_with_dev'],
              monitor_train: bool = False,
              embeddings_in_memory: bool = True,
              checkpoint: bool = False,
              save_final_model: bool = True,
              anneal_with_restarts: bool = False,
              test_mode: bool = False,
              **kwargs
              ) -> dict:
        """训练器类

                Parameters
                ----------
                base_path : str
                    输出路径
                evaluation_metric : Metric
                    评测指标
                learning_rate : float
                    学习率
                max_epochs : int
                    最大epoch数
                anneal_factor : int
                    学习率衰减率
                patience : int
                    不收敛忍耐epoch数量
                anneal_against_train_loss : bool
                    是否以训练loss为学习率衰减指标
                train_with_dev : bool
                   是否把验证集加入训练数据
                monitor_train : bool
                    是否evaluate训练集
                embeddings_in_memory : bool
                    是否把embeddings放到内存
                checkpoint : bool
                    是否开启checkpoint
                save_final_model : bool
                    是否保存最后的model
                """

        if eval_mini_batch_size is None:
            eval_mini_batch_size = mini_batch_size

        if type(base_path) is str:
            base_path = Path(base_path)

        init_log(log, base_path)

        log_line(log)
        log.info(f'Evaluation method: {evaluation_metric.name}')

        loss_txt = init_output_file(base_path, 'loss.tsv')
        with open(loss_txt, 'a') as f:
            f.write(
                f'EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS\t{Metric.tsv_header("TRAIN")}\tDEV_LOSS\t{Metric.tsv_header("DEV")}'
                f'\tTEST_LOSS\t{Metric.tsv_header("TEST")}\n')

        weight_extractor = WeightExtractor(base_path)

        optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, **kwargs)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        anneal_mode = 'min' if anneal_against_train_loss else 'max'

        scheduler = ReduceLROnPlateau(optimizer, factor=anneal_factor,
                                          patience=patience, mode=anneal_mode,
                                          verbose=True)
        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)

        train_data = self.corpus.train

        if train_with_dev:
            train_data.extend(self.corpus.dev)

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []

        try:
            previous_learning_rate = learning_rate

            for epoch in range(0 + self.epoch, max_epochs + self.epoch):
                log_line(log)

                try:
                    bad_epochs = scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    learning_rate = group['lr']

                if learning_rate != previous_learning_rate and anneal_with_restarts and \
                        (base_path / 'best-model.pt').exists():
                    log.info('resetting to best model')
                    self.model.load_from_file(base_path / 'best-model.pt')

                previous_learning_rate = learning_rate

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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
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
                log.info(f'EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.4f} - bad epochs {bad_epochs}')

                dev_metric = None
                dev_loss = '_'

                train_metric = None
                test_metric = None
                if monitor_train:
                    train_metric, train_loss = self._calculate_evaluation_results_for(
                        'TRAIN', self.corpus.train, evaluation_metric, embeddings_in_memory, eval_mini_batch_size)

                dev_metric, dev_loss = self._calculate_evaluation_results_for(
                    'DEV', self.corpus.dev, evaluation_metric, embeddings_in_memory, eval_mini_batch_size)

                test_metric, test_loss = self._calculate_evaluation_results_for(
                    'TEST', self.corpus.test, evaluation_metric, embeddings_in_memory, eval_mini_batch_size,
                    base_path / 'test.tsv')

                with open(loss_txt, 'a') as f:
                    train_metric_str = train_metric.to_tsv() if train_metric is not None else Metric.to_empty_tsv()
                    dev_metric_str = dev_metric.to_tsv() if dev_metric is not None else Metric.to_empty_tsv()
                    test_metric_str = test_metric.to_tsv() if test_metric is not None else Metric.to_empty_tsv()
                    f.write(
                        f'{epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t'
                        f'{train_loss}\t{train_metric_str}\t{dev_loss}\t{dev_metric_str}\t_\t{test_metric_str}\n')

                dev_score = 0.
                if not train_with_dev:
                    if evaluation_metric == EvaluationMetric.MACRO_ACCURACY:
                        dev_score = dev_metric.macro_avg_accuracy()
                    elif evaluation_metric == EvaluationMetric.MICRO_ACCURACY:
                        dev_score = dev_metric.micro_avg_accuracy()
                    elif evaluation_metric == EvaluationMetric.MACRO_F1_SCORE:
                        dev_score = dev_metric.macro_avg_f_score()
                    else:
                        dev_score = dev_metric.micro_avg_f_score()

                    dev_score_history.append(dev_score)
                    dev_loss_history.append(dev_loss.item())

                current_score = train_loss if anneal_against_train_loss else test_metric.micro_avg_f_score()

                scheduler.step(current_score)

                train_loss_history.append(train_loss)

                if checkpoint:
                    self.model.save_checkpoint(base_path / 'checkpoint.pt',
                                               optimizer.state_dict(), scheduler.state_dict(),
                                               epoch + 1, train_loss)

                if current_score == scheduler.best:
                    self.model.save(base_path / 'best-model.pt')

            self.model.save(base_path / 'final-model.pt')

        except KeyboardInterrupt:
            log_line(log)
            log.info('Exiting from training early.')
            log.info('Saving model ...')
            self.model.save(base_path / 'final-model.pt')
            log.info('Done.')

        if self.corpus.test:
            final_score = self.final_test(base_path, embeddings_in_memory, evaluation_metric, eval_mini_batch_size)
        else:
            final_score = 0
            log.info('Test data not provided setting final score to 0')

        return {'test_score': final_score,
                'dev_score_history': dev_score_history,
                'train_loss_history': train_loss_history,
                'dev_loss_history': dev_loss_history}

    def final_test(self,
                   base_path: Path,
                   embeddings_in_memory: bool,
                   evaluation_metric: EvaluationMetric,
                   eval_mini_batch_size: int):

        log_line(log)
        log.info('Testing using best model ...')

        self.model.eval()

        if (base_path / 'best-model.pt').exists():
            if isinstance(self.model, TextClassifier):
                self.model = TextClassifier.load_from_file(base_path / 'best-model.pt')
            if isinstance(self.model, SequenceTagger):
                self.model = SequenceTagger.load_from_file(base_path / 'best-model.pt')

        test_metric, test_loss = self.evaluate(self.model, self.corpus.test, eval_mini_batch_size=eval_mini_batch_size,
                                               embeddings_in_memory=embeddings_in_memory)
        if isinstance(self.model,SequenceTagger):
            log.info(f'MICRO_AVG: acc {test_metric.micro_avg_accuracy()} - f1-score {test_metric.micro_avg_f_score()}')
            log.info(f'MACRO_AVG: acc {test_metric.macro_avg_accuracy()} - f1-score {test_metric.macro_avg_f_score()}')
        elif isinstance(self.model,TextClassifier):
            log.info(f'acc {test_metric.macro_avg_f_score()} - f1-score {test_metric.micro_avg_f_score()}')
        for class_name in test_metric.get_classes():
            log.info(f'{class_name:<10} tp: {test_metric.get_tp(class_name)} - fp: {test_metric.get_fp(class_name)} - '
                     f'fn: {test_metric.get_fn(class_name)} - tn: {test_metric.get_tn(class_name)} - precision: '
                     f'{test_metric.precision(class_name):.4f} - recall: {test_metric.recall(class_name):.4f} - '
                     f'accuracy: {test_metric.accuracy(class_name):.4f} - f1-score: '
                     f'{test_metric.f_score(class_name):.4f}')
        log_line(log)

        # 多个数据集的话，依次测试
        if type(self.corpus) is MultiCorpus:
            for subcorpus in self.corpus.corpora:
                log_line(log)
                self._calculate_evaluation_results_for(subcorpus.name,
                                                       subcorpus.test,
                                                       evaluation_metric,
                                                       embeddings_in_memory,
                                                       eval_mini_batch_size,
                                                       base_path / 'test.tsv')

        # get and return the final test score of best model
        if evaluation_metric == EvaluationMetric.MACRO_ACCURACY:
            final_score = test_metric.macro_avg_accuracy()
        elif evaluation_metric == EvaluationMetric.MICRO_ACCURACY:
            final_score = test_metric.micro_avg_accuracy()
        elif evaluation_metric == EvaluationMetric.MACRO_F1_SCORE:
            final_score = test_metric.macro_avg_f_score()
        else:
            final_score = test_metric.micro_avg_f_score()

        return final_score

    def _calculate_evaluation_results_for(self,
                                          dataset_name: str,
                                          dataset: List[Sentence],
                                          evaluation_metric: EvaluationMetric,
                                          embeddings_in_memory: bool,
                                          eval_mini_batch_size: int,
                                          out_path: Path = None):

        metric, loss = ModelTrainer.evaluate(self.model, dataset, eval_mini_batch_size=eval_mini_batch_size,
                                             embeddings_in_memory=embeddings_in_memory, out_path=out_path)
        if isinstance(self.model,SequenceTagger):
            if evaluation_metric == EvaluationMetric.MACRO_ACCURACY or evaluation_metric == EvaluationMetric.MACRO_F1_SCORE:
                f_score = metric.macro_avg_f_score()
                acc = metric.macro_avg_accuracy()
            else:
                f_score = metric.micro_avg_f_score()
                acc = metric.micro_avg_accuracy()

            log.info(f'{dataset_name:<5}: loss {loss:.8f} - f-score {f_score:.4f} - acc {acc:.4f}')
        elif isinstance(self.model,TextClassifier):
            f_score = metric.micro_avg_f_score()
            acc = metric.macro_avg_f_score()
            log.info(f'{dataset_name:<5}: loss {loss:.8f} - f-score {f_score:.4f} - acc {acc:.4f}')

        return metric, loss

    @staticmethod
    def evaluate(model: nn.Model, data_set: List[Sentence],
                 eval_mini_batch_size: int = 32,
                 embeddings_in_memory: bool = True,
                 out_path: Path = None) -> (
            dict, float):
        if isinstance(model, TextClassifier):
            return ModelTrainer._evaluate_text_classifier(model, data_set, eval_mini_batch_size, embeddings_in_memory,
                                                          out_path)
        elif isinstance(model, SequenceTagger):
            return ModelTrainer._evaluate_sequence_tagger(model, data_set, eval_mini_batch_size, embeddings_in_memory,
                                                          out_path)

    @staticmethod
    def _evaluate_sequence_tagger(model,
                                  sentences: List[Sentence],
                                  eval_mini_batch_size: int = 32,
                                  embeddings_in_memory: bool = True,
                                  out_path: Path = None) -> (dict, float):

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0
            batches = [sentences[x:x + eval_mini_batch_size] for x in range(0, len(sentences), eval_mini_batch_size)]

            metric = Metric('Evaluation')

            lines: List[str] = []
            for batch in batches:
                batch_no += 1

                tags, loss = model.forward_labels_and_loss(batch)

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label('predicted', tag)

                        # append both to file for evaluation
                        eval_line = '{} {} {} {}\n'.format(token.text,
                                                           token.get_tag(model.tag_type).value, tag.value, tag.score)
                        lines.append(eval_line)
                    lines.append('\n')
                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans(model.tag_type)]
                    # make list of predicted tags
                    predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

            eval_loss /= len(sentences)

            if out_path is not None:
                with open(out_path, "w", encoding='utf-8') as outfile:
                    outfile.write(''.join(lines))

            return metric, eval_loss

    @staticmethod
    def _evaluate_text_classifier(model: nn.Model,
                                  sentences: List[Sentence],
                                  eval_mini_batch_size: int = 32,
                                  embeddings_in_memory: bool = False,
                                  out_path: Path = None) -> (dict, float):

        with torch.no_grad():
            eval_loss = 0

            batches = [sentences[x:x + eval_mini_batch_size] for x in
                       range(0, len(sentences), eval_mini_batch_size)]

            metric = Metric('Evaluation')

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
                    eval_line = '{}\t{}\t{}\t{}\n'.format(sentence, true_value, prediction, confidence)
                    lines.append(eval_line)

                for predictions_for_sentence, true_values_for_sentence in zip(predictions_for_batch, true_values_for_batch):
                    ModelTrainer._evaluate_sentence_for_text_classification(metric,
                                                                            available_labels,
                                                                            predictions_for_sentence,
                                                                            true_values_for_sentence)

            eval_loss /= len(sentences)

            if out_path is not None:
                with open(out_path, "w", encoding='utf-8') as outfile:
                    outfile.write(''.join(lines))

            return metric, eval_loss

    @staticmethod
    def _evaluate_sentence_for_text_classification(metric: Metric,
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

    @staticmethod
    def load_from_checkpoint(checkpoint_file: Path, model_type: str, corpus: Corpus, optimizer: Optimizer = SGD):
        if model_type == 'SequenceTagger':
            checkpoint = SequenceTagger.load_checkpoint(checkpoint_file)
            return ModelTrainer(checkpoint['model'], corpus, optimizer, epoch=checkpoint['epoch'],
                                loss=checkpoint['loss'], optimizer_state=checkpoint['optimizer_state_dict'],
                                scheduler_state=checkpoint['scheduler_state_dict'])

        if model_type == 'TextClassifier':
            checkpoint = TextClassifier.load_checkpoint(checkpoint_file)
            return ModelTrainer(checkpoint['model'], corpus, optimizer, epoch=checkpoint['epoch'],
                                loss=checkpoint['loss'], optimizer_state=checkpoint['optimizer_state_dict'],
                                scheduler_state=checkpoint['scheduler_state_dict'])

        raise ValueError('Incorrect model type! Use one of the following: "SequenceTagger", "TextClassifier".')

