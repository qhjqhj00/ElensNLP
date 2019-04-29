import warnings
import logging
from pathlib import Path

import torch.autograd as autograd
import torch.nn

from . import nn
import torch
import os

from lensnlp.Embeddings import TokenEmbeddings
from lensnlp.utilis.data import Dictionary, Sentence, Token, Label

from typing import List, Union
import torch.nn.functional as F

from lensnlp.utilis.training_utils import clear_embeddings
from lensnlp.hyper_parameters import Parameter,device
from lensnlp.Embeddings import WordEmbeddings

log = logging.getLogger('lensnlp')

START_TAG: str = '<START>'
STOP_TAG: str = '<STOP>'

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))


def to_scalar(var):
    """tensor变为普通常量"""
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    """求argmax"""
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    """求log sum exp"""
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    """给一个batch求求argmax"""
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    """给一个batch求log sum exp"""
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list):
    """
    :param tensor_list: 长度不一样的tensors
    :return: 长度一样的tensors
    """
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = torch.zeros(*shape, dtype=torch.long, device=device)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, :lens_[i]] = tensor

    return template, lens_


class SequenceTagger(nn.Model):
    """序列标注模型

            Parameters
            ----------
            hidden_size : int
                隐藏层尺寸
            embeddings : TokenEmbeddings
                词向量
            tag_dictionary : Dictionary
                标签字典
            tag_type : str
                标签类型 如：'ner'
            use_rnn : bool
                是否用rnn
            rnn_layers : int
                rnn的层数
            dropout : float
                dropout率
            word_dropout : float
                word_dropout率
            locked_dropout : float
                locked_dropout率
            Examples
            --------
             >>> from lensnlp.models import SequenceTagger
             >>> from lensnlp.Embeddings import WordEmbeddings
             >>> from lensnlp.utilis.data import Sentence
             >>> sent = Sentence('北京一览群智数据有限公司。')
             >>> emb = WordEmbeddings('cn_glove'
             >>> tagger = SequenceTagger(hidden_size=256,embeddings = emb)
             >>> cn_ner = SequenceTagger.load('cn_s') # 加载预训练模型
             >>> tagger.predict(sent) # 预测
            """

    def __init__(self,
                 hidden_size: int,
                 embeddings: TokenEmbeddings,
                 tag_dictionary: Dictionary,
                 tag_type: str,
                 use_crf: bool = True,
                 use_rnn: bool = True,
                 rnn_layers: int = 1,
                 dropout: float = 0.0,
                 word_dropout: float = 0.05,
                 locked_dropout: float = 0.5,
                 ):


        super(SequenceTagger, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings

        # 设置词典
        self.tag_dictionary: Dictionary = tag_dictionary
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = nn.LockedDropout(locked_dropout)

        rnn_input_dim: int = self.embeddings.embedding_length

        self.relearn_embeddings: bool = True

        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        self.rnn_type = 'LSTM'
        if self.rnn_type in ['LSTM', 'GRU']:

            if self.nlayers == 1:
                self.rnn = getattr(torch.nn, self.rnn_type)(rnn_input_dim, hidden_size,
                                                            num_layers=self.nlayers,
                                                            bidirectional=True)
            else:
                self.rnn = getattr(torch.nn, self.rnn_type)(rnn_input_dim, hidden_size,
                                                            num_layers=self.nlayers,
                                                            dropout=0.5,
                                                            bidirectional=True)

        # final linear map to tag space
        if self.use_rnn:
            self.linear = torch.nn.Linear(hidden_size * 2, len(tag_dictionary))
        else:
            self.linear = torch.nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))

        if self.use_crf:
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size))
            self.transitions.detach()[self.tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
            self.transitions.detach()[:, self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000

        self.to(device)


    def save(self, model_file: Union[str, Path]):
        """保存模型"""
        model_state = {
            'state_dict': self.state_dict(),
            'embeddings': self.embeddings,
            'hidden_size': self.hidden_size,
            'tag_dictionary': self.tag_dictionary,
            'tag_type': self.tag_type,
            'use_crf': self.use_crf,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
            'use_word_dropout': self.use_word_dropout,
            'use_locked_dropout': self.use_locked_dropout,
        }

        torch.save(model_state, str(model_file), pickle_protocol=4)

    def save_checkpoint(self, model_file: Union[str, Path], optimizer_state: dict, scheduler_state: dict, epoch: int,
                        loss: float):
        """保存checkpoint"""
        model_state = {
            'state_dict': self.state_dict(),
            'embeddings': self.embeddings,
            'hidden_size': self.hidden_size,
            'tag_dictionary': self.tag_dictionary,
            'tag_type': self.tag_type,
            'use_crf': self.use_crf,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
            'use_word_dropout': self.use_word_dropout,
            'use_locked_dropout': self.use_locked_dropout,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'epoch': epoch,
            'loss': loss
        }

        torch.save(model_state, str(model_file), pickle_protocol=4)

    @classmethod
    def load_from_file(cls, model_file: Union[str, Path]):
        """加载模型"""
        state = SequenceTagger._load_state(model_file)

        use_dropout = 0.0 if not 'use_dropout' in state.keys() else state['use_dropout']
        use_word_dropout = 0.0 if not 'use_word_dropout' in state.keys() else state['use_word_dropout']
        use_locked_dropout = 0.0 if not 'use_locked_dropout' in state.keys() else state['use_locked_dropout']
        if type(state['embeddings']) is str:
            state['embeddings'] = WordEmbeddings(state['embeddings'])
        model = SequenceTagger(
            hidden_size=state['hidden_size'],
            embeddings=state['embeddings'],
            tag_dictionary=state['tag_dictionary'],
            tag_type=state['tag_type'],
            use_crf=state['use_crf'],
            use_rnn=state['use_rnn'],
            rnn_layers=state['rnn_layers'],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
        )
        model.load_state_dict(state['state_dict'])
        model.eval()

        model.to(device)

        return model

    @classmethod
    def load_checkpoint(cls, model_file: Union[str, Path]):
        """加载checkpoint"""
        state = SequenceTagger._load_state(model_file)
        model = SequenceTagger.load_from_file(model_file)

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
        """加载模型中的属性"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            state = torch.load(str(model_file), map_location=device)
        return state

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        """计算loss"""
        features, lengths, tags = self.forward(sentences)
        return self._calculate_loss(features, lengths, tags)

    def forward_labels_and_loss(self, sentences: Union[List[Sentence], Sentence]) -> (List[List[Label]], torch.tensor):
        """获得预测的标签和计算loss，预测时会调用"""
        with torch.no_grad():
            feature, lengths, tags = self.forward(sentences)
            loss = self._calculate_loss(feature, lengths, tags)
            tags = self._obtain_labels(feature, lengths)
            return tags, loss

    def predict(self, sentences: Union[List[Sentence], Sentence], mini_batch_size=32) -> List[Sentence]:
        """
        预测标签
        :param sentences: 句子
        :param mini_batch_size: batch size
        :return: 有标签的句子
        """
        with torch.no_grad():
            if type(sentences) is Sentence:
                sentences = [sentences]

            filtered_sentences = self._filter_empty_sentences(sentences)

            # 清除每个batch的向量，节省空间
            clear_embeddings(filtered_sentences, also_clear_word_embeddings=True)

            batches = [filtered_sentences[x:x + mini_batch_size] for x in
                       range(0, len(filtered_sentences), mini_batch_size)]

            for batch in batches:
                tags, _ = self.forward_labels_and_loss(batch)  # 预测

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label(self.tag_type, tag)

            return sentences

    def forward(self, sentences: List[Sentence],sort=True):
        """forward，获得特征"""
        self.zero_grad()

        self.embeddings.embed(sentences)

        if sort:
            sentences.sort(key=lambda x: len(x), reverse=True)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        tag_list: List = []
        longest_token_sequence_in_batch: int = lengths[0]

        #初始化0张量
        sentence_tensor = torch.zeros([len(sentences),
                                       longest_token_sequence_in_batch,
                                       self.embeddings.embedding_length],
                                      dtype=torch.float, device=device)

        for s_id, sentence in enumerate(sentences):
            # 用词向量填充
            sentence_tensor[s_id][:len(sentence)] = torch.cat([token.get_embedding().unsqueeze(0)
                                                               for token in sentence], 0)

            # 得到标签序列
            tag_idx: List[int] = [self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                                  for token in sentence]
            # 匹配成张量
            tag = torch.LongTensor(tag_idx).to(device)
            tag_list.append(tag)

        sentence_tensor = sentence_tensor.transpose_(0, 1)

        # 前向传播

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

            rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)

            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)

            if self.use_locked_dropout > 0.0:
                sentence_tensor = self.locked_dropout(sentence_tensor)

        features = self.linear(sentence_tensor)

        return features.transpose_(0, 1), lengths, tag_list

    def _score_sentence(self, feats, tags, lens_):
        """
        gold标签序列的得分，详情请见crf损失函数
        """
        start = torch.LongTensor([self.tag_dictionary.get_idx_for_item(START_TAG)]).to(device)
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.LongTensor([self.tag_dictionary.get_idx_for_item(STOP_TAG)]).to(device)
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]:] = \
                self.tag_dictionary.get_idx_for_item(STOP_TAG)

        score = torch.FloatTensor(feats.shape[0]).to(device)

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(device)

            score[i] = \
                torch.sum(
                    self.transitions[pad_stop_tags[i, :lens_[i] + 1], pad_start_tags[i, :lens_[i] + 1]]
                ) + \
                torch.sum(feats[i, r, tags[i, :lens_[i]]])

        return score

    def _calculate_loss(self, features, lengths, tags) -> float:
        """计算loss"""
        if self.use_crf:
            tags, _ = pad_tensors(tags)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score

            return score.sum()

        else:
            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(features, tags, lengths):
                sentence_feats = sentence_feats[:sentence_length]

                score += torch.nn.functional.cross_entropy(sentence_feats, sentence_tags)

            return score

    def _obtain_labels(self, feature, lengths) -> List[List[Label]]:
        """获得标签"""
        tags = []

        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq = self._viterbi_decode(feats[:length])
            else:
                import torch.nn.functional as F
                softmax = F.softmax(feats[:length], dim=1)
                confidences, tag_seq = torch.max(softmax, 1)
                confidences = [conf.itme() for conf in confidences]
                tag_seq = [tag.item() for tag in tag_seq]
            tags.append([Label(self.tag_dictionary.get_item_for_index(tag), conf)
                         for conf, tag in zip(confidences, tag_seq)])

        return tags

    def _viterbi_decode(self, feats):
        """
        :param feats: forward出来的特征
        :return: 最似然的序列
        预测性能卡在了这，因为每个句子长度不一样，必须单独算。尝试过开多线程，没什么卵用。
        """
        backpointers = []
        backscores = []

        init_vvars = torch.FloatTensor(1, self.tagset_size).to(device).fill_(-10000.)
        init_vvars[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)]
        terminal_var.detach()[self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000.
        terminal_var.detach()[self.tag_dictionary.get_idx_for_item(START_TAG)] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))

        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())

        start = best_path.pop()
        assert start == self.tag_dictionary.get_idx_for_item(START_TAG)
        best_path.reverse()
        return best_scores, best_path

    def _forward_alg(self, feats, lens_):

        """
        :param feats: forward输出的特征
        :param lens_: 句子长度
        :return: 所有可能序列的log sum，详情请见CRF的损失函数
        """

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.)
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float, device=device)

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        transitions = self.transitions.view(
            1,
            self.transitions.shape[0],
            self.transitions.shape[1],
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = \
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2]) + \
                transitions + \
                forward_var[:, i, :][:, :, None].repeat(1, 1, transitions.shape[2]).transpose(2, 1)

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - \
                      max_tag_var[:, :, None].repeat(1, 1, transitions.shape[2])

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + \
                       self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)][None, :].repeat(
                           forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        """
        过滤空的句子
        """
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning('Ignore {} sentence(s) with no tokens.'.format(len(sentences) - len(filtered_sentences)))
        return filtered_sentences

    @staticmethod
    def load(model_file: str):
        """加载预训练模型"""
        if model_file == 'cn_s':
            tagger: SequenceTagger = SequenceTagger.load_from_file(Path(CACHE_ROOT) / 'ner_models/cn/cn_s.pt')
        elif model_file == 'cn_x':
            tagger: SequenceTagger = SequenceTagger.load_from_file(Path(CACHE_ROOT) / 'ner_models/cn/cn_x.pt')
        elif model_file == 'en_s':
            tagger: SequenceTagger = SequenceTagger.load_from_file(Path(CACHE_ROOT) / 'ner_models/en/en_s.pt')
        elif model_file == 'en_x':
            tagger: SequenceTagger = SequenceTagger.load_from_file(Path(CACHE_ROOT) / 'ner_models/en/en_x.pt')
        elif model_file == 'uy_s':
            tagger: SequenceTagger = SequenceTagger.load_from_file(Path(CACHE_ROOT) / 'ner_models/uy/uy_s.pt')
        elif model_file == 'cn_5':
            tagger: SequenceTagger = SequenceTagger.load_from_file(Path(CACHE_ROOT) / 'ner_models/cn/cn_5.pt')
        else:
            try:
                tagger: SequenceTagger = SequenceTagger.load_from_file(Path(CACHE_ROOT) / model_file)
            except NameError:
                print('No such a model!')
        return tagger
