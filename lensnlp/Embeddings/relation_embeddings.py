from .base import DocumentEmbeddings,TokenEmbeddings
from .word_embeddings import StackedEmbeddings
import logging
from typing import List, Union

import torch
from lensnlp.utilis.data import Sentence, Token, Dictionary
from lensnlp.hyper_parameters import device

log = logging.getLogger('lensnlp')


class Relation_Embeddings(DocumentEmbeddings):
    """
    针对关系抽取任务，添加句子级词向量以及对应的位置向量
    """
    def __init__(self, embeddings: List[TokenEmbeddings],
                 pos_dict: Dictionary,
                 pos_embedding_length: int = 50,
                 word_dropout: float = 0.0,
                 ):

        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.pos_dict = pos_dict
        self.pos_dict.add_item('padding')

        self.__embedding_length: int = self.embeddings.embedding_length
        self.__pos_embedding_length: int = pos_embedding_length

        self.pos_embedding = torch.nn.Embedding(
            len(self.pos_dict.item2idx), self.pos_embedding_length
        )

        self.name: str = f're'
        self.to(device)

        self.use_word_dropout: bool = word_dropout > 0.0

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @property
    def pos_embedding_length(self):
        return self.__pos_embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """
        :param sentences
        :return: 句子级别的向量，由词向量求得
        """
        everything_embedded: bool = True

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        all_sentence_p1 = []
        all_sentence_p2 = []
        lengths: List[int] = []

        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []
            p1_embeddings = []
            p2_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                word_embeddings.append(token.get_embedding().unsqueeze(0))
                p1_embeddings.append(self.pos_dict.get_idx_for_item(token.relative_position['p1']))
                p2_embeddings.append(self.pos_dict.get_idx_for_item(token.relative_position['p2']))

            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.zeros(self.embeddings.embedding_length,
                                dtype=torch.float).unsqueeze(0)
                )

                p1_embeddings.append(self.pos_dict.get_idx_for_item('padding'))
                p2_embeddings.append(self.pos_dict.get_idx_for_item('padding'))

            p1_embeddings = self.pos_embedding(torch.LongTensor(p1_embeddings).to(device))
            p2_embeddings = self.pos_embedding(torch.LongTensor(p2_embeddings).to(device))

            word_embeddings_tensor = torch.cat(word_embeddings, 0).to(device)
            # p1_embeddings_tensor = torch.cat(p1_embeddings, 0).to(device)
            # p2_embeddings_tensor = torch.cat(p2_embeddings, 0).to(device)

            sentence_states = word_embeddings_tensor

            # 加到一个list中
            all_sentence_tensors.append(sentence_states.unsqueeze(0))
            all_sentence_p1.append(p1_embeddings.unsqueeze(0))
            all_sentence_p2.append(p2_embeddings.unsqueeze(0))

        # 得到batch的特征
        sentence_tensor = torch.cat(all_sentence_tensors, 0)
        sentence_p1_tensor = torch.cat(all_sentence_p1,0)
        sentence_p2_tensor = torch.cat(all_sentence_p2,0)

        return sentence_tensor, sentence_p1_tensor, sentence_p2_tensor

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass

