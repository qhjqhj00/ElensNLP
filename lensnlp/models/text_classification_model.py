import warnings
import logging
from pathlib import Path
from typing import List, Union
import os

import torch
import torch.nn.functional as F

from lensnlp.hyper_parameters import Parameter,device
from lensnlp.models import nn
from lensnlp.embeddings import TokenEmbeddings, DocumentEmbeddings,DocumentRNNEmbeddings,WordEmbeddings
from lensnlp.utils.data import Dictionary, Sentence, Label
from lensnlp.utils.training_utils import convert_labels_to_one_hot, clear_embeddings


log = logging.getLogger('lensnlp')

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))


class TextClassifier(nn.Model):
    """序列标注模型

            Parameters
            ----------
            document_embeddings : DocumentEmbeddings
                词向量
            label_dictionary : Dictionary
                label的匹配词典
            multi_label : bool
                是否采用多标签
            Examples
            --------
             >>> from lensnlp.models import TextClassifier
             >>> from lensnlp.embeddings import WordEmbeddings,DocumentPoolEmbeddings
             >>> from lensnlp.utils.data import Sentence
             >>> sent = Sentence('北京一览群智数据有限公司。')
             >>> emb_list = [WordEmbeddings('cn_glove')]
             >>> doc_emb = DocumentPoolEmbeddings(emb_list)
             >>> clf = TextClassifier(document_embeddings=doc_emb,label_dictionary=None)
             >>> cn_clf = TextClassifier.load('cn_clf') # 加载预训练模型
             >>> cn_clf.predict(sent) # 预测
            """


    def __init__(self,
                 document_embeddings: DocumentEmbeddings,
                 label_dictionary: Dictionary,
                 multi_label: bool,):

        super(TextClassifier, self).__init__()

        self.document_embeddings = document_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.multi_label = multi_label

        self.document_embeddings= document_embeddings

        self.decoder = torch.nn.Linear(self.document_embeddings.embedding_length, len(self.label_dictionary))

        self._init_weights()

        if multi_label:
            self.loss_function = torch.nn.BCELoss()
        else:
            self.loss_function = torch.nn.CrossEntropyLoss()
        self.to(device)

    def _init_weights(self):
        """初始化权重"""
        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, sentences) -> List[List[float]]:
        """forward获取特征"""
        self.document_embeddings.embed(sentences)

        text_embedding_list = [sentence.get_embedding().unsqueeze(0) for sentence in sentences]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(device)

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def save(self, model_file: Union[str, Path]):
        """
        存模型
        """
        model_state = {
            'state_dict': self.state_dict(),
            'document_embeddings': self.document_embeddings,
            'label_dictionary': self.label_dictionary,
            'multi_label': self.multi_label}
        torch.save(model_state, str(model_file), pickle_protocol=4)

    def save_checkpoint(self, model_file: Union[str, Path], optimizer_state: dict, scheduler_state: dict, epoch: int, loss: float):
        """
        存断点
        """
        model_state = {
            'state_dict': self.state_dict(),
            'document_embeddings': self.document_embeddings,
            'label_dictionary': self.label_dictionary,
            'multi_label': self.multi_label,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'epoch': epoch,
            'loss': loss
        }
        torch.save(model_state, str(model_file), pickle_protocol=4)

    @classmethod
    def load_from_file(cls, model_file: Union[str, Path]):
        """
        :param model_file: 模型地址
        :return: 加载好的模型
        """
        state = TextClassifier._load_state(model_file)
        if type(state['document_embeddings']) is dict:
            if state['document_embeddings']['type'] == 'rnn' and state['document_embeddings']['emb'] == 'cn_glove':
                state['document_embeddings'] = DocumentRNNEmbeddings([WordEmbeddings('cn_glove')])
            elif state['document_embeddings']['type'] == 'bilstm' and state['document_embeddings']['emb'] == 'cn_glove':
                state['document_embeddings'] = DocumentRNNEmbeddings([WordEmbeddings('cn_glove')],
                                                                     bidirectional=True, use_attention=True)
        model = TextClassifier(
            document_embeddings=state['document_embeddings'],
            label_dictionary=state['label_dictionary'],
            multi_label=state['multi_label']
        )
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(device)

        return model

    @classmethod
    def load_checkpoint(cls, model_file: Union[str, Path]):
        """加载断电模型"""
        state = TextClassifier._load_state(model_file)
        model = TextClassifier.load_from_file(model_file)

        epoch = state['epoch'] if 'epoch' in state else None
        loss = state['loss'] if 'loss' in state else None
        optimizer_state_dict = state['optimizer_state_dict'] if 'optimizer_state_dict' in state else None
        scheduler_state_dict = state['scheduler_state_dict'] if 'scheduler_state_dict' in state else None

        return {
            'model': model, 'epoch': epoch, 'loss': loss,
            'optimizer_state_dict': optimizer_state_dict, 'scheduler_state_dict': scheduler_state_dict
        }

    @classmethod
    def _load_state(cls, model_file: Union[str, Path]):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            state = torch.load(str(model_file), map_location=device)
            return state

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        scores = self.forward(sentences)
        return self._calculate_loss(scores, sentences)

    def forward_labels_and_loss(self, sentences: Union[Sentence, List[Sentence]],
                                get_prob=False) -> (List[List[Label]], torch.tensor):
        scores = self.forward(sentences)
        labels = self._obtain_labels(scores,get_prob)
        loss = self._calculate_loss(scores, sentences)
        return labels, loss

    def predict(self, sentences: Union[Sentence, List[Sentence]],
                mini_batch_size: int = 32, get_prob = False) -> List[Sentence]:
        """
        预测
        输入为 Sentence 数量不限
        返回 Sentence，标签存入对应的位置
        mini_batch_size为每个batch预测的数量
        """
        with torch.no_grad():
            if type(sentences) is Sentence:
                sentences = [sentences]

            filtered_sentences = self._filter_empty_sentences(sentences)

            batches = [filtered_sentences[x:x + mini_batch_size] for x in range(0, len(filtered_sentences), mini_batch_size)]

            for batch in batches:
                scores = self.forward(batch)
                predicted_labels = self._obtain_labels(scores,get_prob)

                for (sentence, labels) in zip(batch, predicted_labels):
                    sentence.labels = labels

                clear_embeddings(batch)

            return sentences

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning('Ignore {} sentence(s) with no tokens.'.format(len(sentences) - len(filtered_sentences)))
        return filtered_sentences

    def _calculate_loss(self, scores: List[List[float]], sentences: List[Sentence]) -> float:

        if self.multi_label:
            return self._calculate_multi_label_loss(scores, sentences)

        return self._calculate_single_label_loss(scores, sentences)

    def _obtain_labels(self, scores: List[List[float]], get_prob=False) -> List[List[Label]]:
        """
        获得文本标签
        """

        if self.multi_label:
            return [self._get_multi_label(s) for s in scores]

        elif get_prob:
            return [self._get_prob(s) for s in scores]

        return [self._get_single_label(s) for s in scores]

    def _get_multi_label(self, label_scores) -> List[Label]:
        labels = []

        sigmoid = torch.nn.Sigmoid()

        results = list(map(lambda x: sigmoid(x), label_scores))
        for idx, conf in enumerate(results):
            if conf > 0.5:
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))

        return labels

    def _get_prob(self, label_scores):
        label_scores = F.softmax(label_scores,dim=0)
        conf, idx = torch.max(label_scores, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())
        return [Label(label, conf.item())]

    def _get_single_label(self, label_scores) -> List[Label]:
        conf, idx = torch.max(label_scores, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [Label(label, conf.item())]

    def _calculate_multi_label_loss(self, label_scores, sentences: List[Sentence]) -> float:
        sigmoid = torch.nn.Sigmoid()
        return self.loss_function(sigmoid(label_scores), self._labels_to_one_hot(sentences))

    def _calculate_single_label_loss(self, label_scores, sentences: List[Sentence]) -> float:
        return self.loss_function(label_scores, self._labels_to_indices(sentences))

    def _labels_to_one_hot(self, sentences: List[Sentence]):
        label_list = [sentence.get_label_names() for sentence in sentences]
        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0).to(device)
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [
            torch.LongTensor([self.label_dictionary.get_idx_for_item(label.value) for label in sentence.labels])
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0).to(device)

        return vec

    @staticmethod
    def load(model_file: str):
        if model_file == 'cn_clf':
            classifier: TextClassifier = TextClassifier.load_from_file(Path(CACHE_ROOT) / 'clf_models/cn/cn_clf.pt')
        elif model_file == 'cn_emo':
            classifier: TextClassifier = TextClassifier.load_from_file(Path(CACHE_ROOT) / 'clf_models/cn/cn_emo.pt')
        elif model_file == 'uy_clf':
            classifier: TextClassifier = TextClassifier.load_from_file(Path(CACHE_ROOT) / 'clf_models/uy/uy_clf.pt')
        elif model_file == 'uy_emo':
            classifier: TextClassifier = TextClassifier.load_from_file(Path(CACHE_ROOT) / 'clf_models/uy/uy_emo.pt')
        elif model_file == 'en_clf':
            classifier: TextClassifier = TextClassifier.load_from_file(Path(CACHE_ROOT) / 'clf_models/en/en_clf.pt')
        elif model_file == 'en_emo':
            classifier: TextClassifier = TextClassifier.load_from_file(Path(CACHE_ROOT) / 'clf_models/en/en_emo.pt')
        else:
            try:
                classifier: TextClassifier = TextClassifier.load_from_file(Path(CACHE_ROOT) / model_file)
            except NameError('specify a model!'):
                raise
        return classifier
