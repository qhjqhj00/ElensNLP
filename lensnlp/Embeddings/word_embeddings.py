from lensnlp.Embeddings import TokenEmbeddings

import logging
from typing import List, Union, Dict
from pathlib import Path
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

import numpy as np
import torch
from bpemb import BPEmb
from abc import abstractmethod
import os

from lensnlp.utilis.data import Token, Sentence

from pytorch_pretrained_bert import BertTokenizer, BertModel

from lensnlp.hyper_parameters import device

log = logging.getLogger('qhj')

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))


class StackedEmbeddings(TokenEmbeddings):
    """多种词向量叠加"""

    def __init__(self, embeddings: List[TokenEmbeddings], detach: bool = True):
        super().__init__()

        self.embeddings = embeddings

        # 各种词向量按torch模块方式加入
        for i, embedding in enumerate(embeddings):
            self.add_module('list_embedding_{}'.format(i), embedding)

        self.detach: bool = detach
        self.name: str = 'Stack'
        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True):
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'


class WordEmbeddings(TokenEmbeddings):
    """基本的静态词向量，如glove fasttext等"""

    def __init__(self, embeddings: str):

        if embeddings.lower() == 'en_glove':
            embeddings = Path(CACHE_ROOT) / 'language_model/en_glove_300d'
            self.precomputed_word_embeddings = KeyedVectors.load(str(embeddings))

        elif embeddings.lower() == 'cn_glove':
            embeddings = Path(CACHE_ROOT) / 'language_model/cn_glove_300d'
            self.precomputed_word_embeddings = KeyedVectors.load(str(embeddings))

        elif embeddings.lower() == 'cn_fasttext':
            embeddings = Path(CACHE_ROOT) / 'language_model/zh'
            self.precomputed_word_embeddings = FastText.load_fasttext_format(str(embeddings))
        else:
            raise ValueError('Please specify another embeddings!')

        self.name: str = str(embeddings)
        self.static_embeddings = True
        self.__embedding_length: int = self.precomputed_word_embeddings['的'].shape[0]
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        embed_dict = self.precomputed_word_embeddings

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                if token.text in embed_dict:
                    word_embedding = embed_dict[token.text]
                else:
                    word_embedding = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), self.__embedding_length)

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class FlairEmbeddings(TokenEmbeddings):
    """
    字符级 Contextualized Embedding，为每个单词根据前后文生成向量。

    """
    def __init__(self, model: str, trans: str = None, detach: bool = True,
            use_cache: bool = True, cache_directory: Path = None):

        super().__init__()

        if model.lower() == 'uy_forward_small':
            model_path = Path(CACHE_ROOT) / 'language_model/uyghur_forward_small.pt'
        elif model.lower() == 'uy_backward_small':
            model_path = Path(CACHE_ROOT) / 'language_model/uyghur_backward_small.pt'
        elif model.lower() == 'uy_forward_large':
            model_path = Path(CACHE_ROOT) / 'language_model/uyghur_forward_large.pt'
        elif model.lower() == 'uy_backward_large':
            model_path = Path(CACHE_ROOT) / 'language_model/uyghur_backward_large.pt'
        elif model.lower() == 'cn_forward_large':
            model_path = Path(CACHE_ROOT) / 'language_model/cn_forward_large.pt'
        elif model.lower() == 'cn_backward_large':
            model_path = Path(CACHE_ROOT) / 'language_model/cn_backward_large.pt'
        elif model.lower() == 'cn_forward_small':
            model_path = Path(CACHE_ROOT) / 'language_model/cn_forward_small.pt'
        elif model.lower() == 'cn_backward_small':
            model_path = Path(CACHE_ROOT) / 'language_model/cn_backward_small.pt'
        elif model.lower() == 'en_forward_small':
            model_path = Path(CACHE_ROOT) / 'language_model/en_forward_small.pt'
        elif model.lower() == 'en_backward_small':
            model_path = Path(CACHE_ROOT) / 'language_model/en_backward_small.pt'
        elif model.lower() == 'en_forward_large':
            model_path = Path(CACHE_ROOT) / 'language_model/en_forward_large.pt'
        elif model.lower() == 'en_backward_large':
            model_path = Path(CACHE_ROOT) / 'language_model/en_backward_large.pt'
        elif not Path(model).exists():
            raise ValueError(f'The given model "{model}" is not available or is not a valid path.')

        self.name = str(model)
        self.static_embeddings = detach

        self.trans = trans

        from lensnlp.models import LanguageModel
        self.lm = LanguageModel.load_language_model(Path(model_path))
        self.detach = detach

        self.is_forward_lm: bool = self.lm.is_forward_lm

        self.cache = None
        if use_cache:
            cache_path = Path(f'{self.name}-tmp-cache.sqllite') if not cache_directory else \
                Path(cache_directory) / f'{self.name}-tmp-cache.sqllite'
            from sqlitedict import SqliteDict
            self.cache = SqliteDict(str(cache_path), autocommit=True)
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token('hello',lang=self.trans))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0].get_token(1).get_embedding())

        # 模型权重不更新
        self.eval()

    def train(self, mode=True):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = None
        return state

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # 先从缓存中找
        if 'cache' in self.__dict__ and self.cache is not None:

            # 尝试从缓存中找到所有的向量
            all_embeddings_retrieved_from_cache: bool = True
            for sentence in sentences:
                key = sentence.to_tokenized_string()
                embeddings = self.cache.get(key)

                if not embeddings:
                    all_embeddings_retrieved_from_cache = False
                    break
                else:
                    for token, embedding in zip(sentence, embeddings):
                        token.set_embedding(self.name, torch.FloatTensor(embedding))

            if all_embeddings_retrieved_from_cache:
                return sentences

        with torch.no_grad():

            text_sentences = [sentence.to_tokenized_string(self.trans) for sentence in sentences]

            longest_character_sequence_in_batch: int = len(max(text_sentences, key=len))

            # 补全
            sentences_padded: List[str] = []
            append_padded_sentence = sentences_padded.append

            end_marker = ' '
            extra_offset = 1
            for sentence_text in text_sentences:
                pad_by = longest_character_sequence_in_batch - len(sentence_text)
                if self.is_forward_lm:
                    padded = '\n{}{}{}'.format(sentence_text, end_marker, pad_by * ' ')
                    append_padded_sentence(padded)
                else:
                    padded = '\n{}{}{}'.format(sentence_text[::-1], end_marker, pad_by * ' ')
                    append_padded_sentence(padded)

            # 获得特征
            all_hidden_states_in_lm = self.lm.get_representation(sentences_padded, self.detach)

            # 前向模型获得单词最后一个作为特征，反向模型获得第一个
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string(self.trans)

                offset_forward: int = extra_offset
                offset_backward: int = len(sentence_text) + extra_offset

                for token in sentence.tokens:
                    token: Token = token
                    if self.trans == 'UY':
                        offset_forward += len(token.latin)
                    elif self.trans == 'PY':
                        offset_forward += len(token.pinyin)
                    else:
                        offset_forward += len(token.text)

                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    offset_forward += 1
                    offset_backward -= 1
                    if self.trans == 'UY':
                        offset_backward -= len(token.latin)
                    elif self.trans == 'PY':
                        offset_backward -= len(token.pinyin)
                    else:
                        offset_backward -= len(token.text)

                    token.set_embedding(self.name, embedding)

        if 'cache' in self.__dict__ and self.cache is not None:
            for sentence in sentences:
                self.cache[sentence.to_tokenized_string(self.trans)] = [token._embeddings[self.name].tolist()
                                                                        for token in sentence]

        return sentences

    def __str__(self):
        return self.name


class BPEmbSerializable(BPEmb):
    """
    初始化一个BPEmb的模型
    """
    def __getstate__(self):
        state = self.__dict__.copy()
        state['spm_model_binary'] = open(self.model_file, mode='rb').read()
        state['spm'] = None
        return state

    def __setstate__(self, state):
        from bpemb.util import sentencepiece_load
        model_file = self.model_tpl.format(lang=state['lang'], vs=state['vs'])
        self.__dict__ = state

        self.cache_dir: Path = Path(os.path.expanduser(os.path.join('~', '.uy'))) / 'embeddings'
        if 'spm_model_binary' in self.__dict__:
            if not os.path.exists(self.cache_dir / state['lang']):
                os.makedirs(self.cache_dir / state['lang'])
            self.model_file = self.cache_dir / model_file
            with open(self.model_file, 'wb') as out:
                out.write(self.__dict__['spm_model_binary'])
        else:
            self.model_file = self._load_file(model_file)

        state['spm'] = sentencepiece_load(self.model_file)


class BytePairEmbeddings(TokenEmbeddings):
    """
    用Bpemb预训练语言模型来得到词向量
    """
    def __init__(self, language: str, dim: int = 50, syllables: int = 100000,
                 cache_dir=Path(os.path.expanduser(os.path.join('~', '.uy'))) / 'embeddings'):
        self.name: str = f'bpe-{language}-{syllables}-{dim}'
        self.static_embeddings = True
        self.embedder = BPEmbSerializable(lang=language, vs=syllables, dim=dim, cache_dir=cache_dir)

        self.__embedding_length: int = self.embedder.emb.vector_size * 2
        super().__init__()

    @property
    def embedding_length(self) -> int:
        """
        :return: 向量长度
        """
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        :param sentences: 一个batch的句子
        :return: embedding全部存进Token.embeddings中了
        """
        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                word = token.text

                if word.strip() == '':
                    # 空的不给向量
                    token.set_embedding(self.name, torch.zeros(self.embedding_length, dtype=torch.float))
                else:
                    embeddings = self.embedder.embed(word.lower())
                    embedding = np.concatenate((embeddings[0], embeddings[len(embeddings) - 1]))
                    token.set_embedding(self.name, torch.tensor(embedding, dtype=torch.float))

        return sentences

    def __str__(self):
        return self.name


class BertEmbeddings(TokenEmbeddings):

    def __init__(self,
                 bert_model_or_path: str = 'bert-base-uncased',
                 layers: str = '-1,-2,-3,-4',
                 pooling_operation: str = 'first'):
        """
        :param bert_model_or_path: 模型的名字
        :param layers: 选择用那几层
        :param pooling_operation: 词分成多个部分的时候的pool操作方式
        """
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
        self.model = BertModel.from_pretrained(bert_model_or_path)
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.name = str(bert_model_or_path)
        self.static_embeddings = True

    class BertInputFeatures(object):
        """转换成bert格式的特征输入"""

        def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, token_subtoken_count):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(self, sentences, max_sequence_length: int) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0:(max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            # 做padding
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(BertEmbeddings.BertInputFeatures(
                unique_id=sentence_index,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                token_subtoken_count=token_subtoken_count))

        return features

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # 找到最长句
        longest_sentence_in_batch: int = len(
            max([self.tokenizer.tokenize(sentence.to_tokenized_string()) for sentence in sentences], key=len))

        # 做mapping
        features = self._convert_sentences_to_features(sentences, longest_sentence_in_batch)
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(device)
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(device)

        # forward bert模型
        self.model.to(device)
        self.model.eval()
        all_encoder_layers, _ = self.model(all_input_ids, token_type_ids=None, attention_mask=all_input_masks)

        with torch.no_grad():

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu()[sentence_index]
                        all_layers.append(layer_output[token_index])

                    subtoken_embeddings.append(torch.cat(all_layers))

                token_idx = 0
                for token in sentence:
                    # 把多个block的输出连接起来
                    token_idx += 1

                    if self.pooling_operation == 'first':
                        # 用第一个词的输出
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # 或者求个均值
                        embeddings = subtoken_embeddings[token_idx:token_idx + feature.token_subtoken_count[token.idx]]
                        embeddings = [embedding.unsqueeze(0) for embedding in embeddings]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """返回词向量的长度"""
        return len(self.layer_indexes) * self.model.config.hidden_size
