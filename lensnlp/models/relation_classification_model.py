import warnings
import logging
from pathlib import Path

from . import nn
import torch
from typing import List, Union
from lensnlp.Embeddings import *
from lensnlp.utilis.data import Dictionary, Sentence, Label, Token
from lensnlp.hyper_parameters import Parameter,device
from lensnlp.utilis.training_utils import convert_labels_to_one_hot, clear_embeddings
import os

import torch.nn.functional as F
from lensnlp.modules import attention

log = logging.getLogger('lensnlp')

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))


class RelationExtraction(nn.Model):

    def __init__(self,
                 embeddings: RelationEmbeddings,
                 label_dictionary,
                 hidden_size: int = 300,
                 rnn_layer: int = 1,
                 attention_size: int =50,
                 dropout: float = 0.3,
                 word_dropout: float = 0.3,
                 attention_dropout: float = 0.5,
                 pos_embedding_dim: int = 50,
                 ):
        """
        :param embeddings: embeddings for Relation extraction
        :param label_dictionary: dictionary of label
        :param hidden_size:output size of RNN
        :param rnn_layer:depth of Rnn
        :param attention_size:head of attention
        :param dropout:rate of dropout
        :param word_dropout:
        :param locked_dropout:
        :param pos_embedding_dim:dimension of position embeddings
        """
        super(RelationExtraction,self).__init__()

        self.embeddings = embeddings

        self.num_classes: int = len(label_dictionary)
        self.label_dictionary = label_dictionary

        self.attention_size = attention_size
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = nn.WordDropout(word_dropout)

        if attention_dropout > 0.0:
            self.attention_dropout = torch.nn.Dropout(attention_dropout)

        self.loss_function = torch.nn.CrossEntropyLoss()

        self.pos_dim = pos_embedding_dim

        rnn_input_dim: int = self.embeddings.embedding_length

        self.bilstm = torch.nn.LSTM(rnn_input_dim, hidden_size,
                                    num_layers=rnn_layer, bidirectional=True, batch_first=True)

        self.decoder = torch.nn.Linear(2*hidden_size, self.num_classes)

        self.att_l1 = torch.nn.Linear(8*hidden_size, self.attention_size)
        self.att_l2 = torch.nn.Linear(2*hidden_size+2*self.pos_dim, self.attention_size)

        self.attention1 = attention.MultiHeadedAttention(4, self.embeddings.embedding_length)
        self.attention2 = attention.MultiHeadedAttention(self.attention_size, hidden_size)

        self.to(device)

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.xavier_uniform_(self.att_l1.weight)
        torch.nn.init.xavier_uniform_(self.att_l2.weight)

    def forward(self, sentences):

        sentence_tensor, sentence_p1_tensor, sentence_p2_tensor = self.embeddings.embed(sentences)

        if self.word_dropout is not None:
            sentence_tensor = self.word_dropout(sentence_tensor)

        e1_pos = [sen.entity['e1'] for sen in sentences]
        e2_pos = [sen.entity['e2'] for sen in sentences]

        self_attn = self.attention1.forward(sentence_tensor,sentence_tensor,sentence_tensor)
        rnn_outputs, hidden = self.bilstm(self_attn)

        self.dropout(rnn_outputs)

        output, alphas, e1_alphas, e2_alphas = self.entity_aware_attention(rnn_outputs, e1_pos, e2_pos,
                                                                           sentence_p1_tensor, sentence_p2_tensor)

        h_drop = self.attention_dropout(output)

        output = self.decoder(h_drop)

        return output

    def entity_aware_attention(self, rnn_outputs, e1_pos, e2_pos,
                               sentence_p1_tensor, sentence_p2_tensor):
        seq_len = rnn_outputs.shape[1]  # fixed at run-time
        hidden_size = rnn_outputs.shape[2]  # fixed at compile-time #Is there any difference?
        latent_size = hidden_size

        # Latent Relation Variable based on Entities
        e1_h = self.extract_entity_states(rnn_outputs, e1_pos).to(device)
        e2_h = self.extract_entity_states(rnn_outputs, e2_pos).to(device)  # (batch, hidden)

        e1_type, e2_type, e1_alphas, e2_alphas = self.latent_type_attention(e1_h, e2_h,
                                                                            num_type=3, latent_size=latent_size)

        e1_h = torch.cat((e1_h, e1_type), dim=-1).to(device)  # (batch, hidden+latent)
        e2_h = torch.cat((e2_h, e2_type), dim=-1).to(device)

        e_h = self.att_l1(torch.cat((e1_h, e2_h), -1)).unsqueeze(1).to(device)

        e_h = e_h.repeat(1, seq_len, 1)

        v = self.att_l2(torch.cat((rnn_outputs, sentence_p1_tensor, sentence_p2_tensor), -1)).to(device)

        v = torch.tanh(torch.add(v, e_h))

        u_omega = torch.randn(self.attention_size, requires_grad=True).unsqueeze(-1).to(device)

        attn = torch.matmul(v, u_omega).squeeze().to(device)
        alphas = F.softmax(attn, dim=1).to(device)
        output = torch.sum(rnn_outputs * alphas.unsqueeze(-1), 1).to(device)

        return output, alphas, e1_alphas, e2_alphas

    @staticmethod
    def latent_type_attention(e1, e2, num_type, latent_size):
        """
        :param e1:embeddings from rnn of entity 1
        :param e2: embeddings from rnn of entity 2
        :param num_type:
        :param latent_size: # of latent att cell
        :return:
        """
        # Latent Entity Type Vectors
        latent_type = torch.randn(num_type, latent_size, requires_grad=True).to(device)

        e1_sim = torch.matmul(e1, torch.transpose(latent_type, 0, 1))  # (batch, num_type)
        e1_alphas = F.softmax(e1_sim, dim=1)  # (batch, num_type)
        e1_type = torch.matmul(e1_alphas, latent_type)  # (batch, hidden)

        e2_sim = torch.matmul(e2, torch.transpose(latent_type, 0, 1))  # (batch, num_type)
        e2_alphas = F.softmax(e2_sim, dim=1)  # (batch, num_type)
        e2_type = torch.matmul(e2_alphas, latent_type)  # (batch, hidden)

        return e1_type, e2_type, e1_alphas, e2_alphas

    @staticmethod
    def extract_entity_states(rnn_out, ent_pos):
        """
        :param rnn out: output of rnn
        :param e:index of entities in sentences
        :return: entities embeddings
        """
        entity_states = []
        for index,pos in enumerate(ent_pos):
            entity_states.append(rnn_out[index, int(pos), :].unsqueeze(0))
        return torch.cat(entity_states, 0)  # (batch, hidden)

    # Calculate mean cross-entropy loss
    def forward_loss(self,sentences):
        scores = self.forward(sentences)
        return self._calculate_loss(scores, sentences)

    def forward_labels_and_loss(self, sentences: Union[Sentence, List[Sentence]]) -> (List[List[Label]], torch.Tensor):
        scores = self.forward(sentences)
        labels = self._obtain_labels(scores)
        loss = self._calculate_loss(scores, sentences)
        return labels, loss

    def _calculate_loss(self, scores: List[List[float]], sentences: List[Sentence]) -> float:

        return self._calculate_single_label_loss(scores, sentences)

    def _obtain_labels(self, scores: List[List[float]]) -> List[List[Label]]:
        """
        获得文本标签
        """
        return [self._get_single_label(s) for s in scores]

    def _get_single_label(self, label_scores) -> List[Label]:
        conf, idx = torch.max(label_scores, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [Label(label, conf.item())]

    def _calculate_single_label_loss(self, label_scores, sentences: List[Sentence]) -> float:
        return self.loss_function(label_scores, self._labels_to_indices(sentences))

    def _labels_to_one_hot(self, sentences: List[Sentence]):
        label_list = [sentence.get_label_names() for sentence in sentences]
        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = (torch.FloatTensor(l).unsqueeze(0) for l in one_hot)
        one_hot = torch.cat(one_hot, 0).to(device)
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [
            torch.LongTensor([self.label_dictionary.get_idx_for_item(label.value) for label in sentence.labels])
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0).to(device)

        return vec

    def save(self, model_file: Union[str, Path]):
        """
        存模型
        """
        model_state = {
            'state_dict': self.state_dict(),
            'relation_embeddings': self.embeddings,
            'label_dictionary': self.label_dictionary}
        torch.save(model_state, str(model_file), pickle_protocol=4)

    def save_checkpoint(self, model_file: Union[str, Path], optimizer_state: dict, scheduler_state: dict, epoch: int, loss: float):
        """
        存断点
        """
        model_state = {
            'state_dict': self.state_dict(),
            'relation_embeddings': self.embeddings,
            'label_dictionary': self.label_dictionary,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'epoch': epoch,
            'loss': loss
        }
        torch.save(model_state, str(model_file), pickle_protocol=4)

    def predict(self, sentences: Union[Sentence, List[Sentence]],
                mini_batch_size: int = 32) -> List[Sentence]:
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
                predicted_labels = self._obtain_labels(scores)

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


    @classmethod
    def load_from_file(cls, model_file: Union[str, Path]):
        """
        :param model_file: 模型地址
        :return: 加载好的模型
        """
        state = RelationExtraction._load_state(model_file)
        if type(state['relation_embeddings']) is str:
            state['relation_embeddings'] = Relation_Embeddings(state['relation_embeddings'])
        model = RelationExtraction(
            embeddings=state['relation_embeddings'],
            label_dictionary=state['label_dictionary'],
        )
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(device)

        return model

    @classmethod
    def load_checkpoint(cls, model_file: Union[str, Path]):
        state = RelationExtraction._load_state(model_file)
        model = RelationExtraction.load_from_file(model_file)

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

    def load(model_file: str):
        if model_file == 'test_re':
            classifier: RelationExtraction = RelationExtraction.load_from_file(Path(CACHE_ROOT) / 're_models/best-mdoel.pt')
        else:
            try:
                classifier: RelationExtraction = RelationExtraction.load_from_file(Path(CACHE_ROOT) / model_file)
            except NameError('specify a model!'):
                raise
        return classifier