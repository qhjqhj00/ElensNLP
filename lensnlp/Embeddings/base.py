from abc import abstractmethod
from typing import List, Union
import torch

from lensnlp.utilis.data import Sentence


class Embeddings(torch.nn.Module):
    """所有Embedding类的基础类，Embedding类都继承这个类"""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """返回向量的长度"""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        """返回向量的类别"""
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """
        此函数给输入的句子加入向量
        :param sentences: 单个Sentence或者Sentence的列表
        :return: 储存好向量的sentences
        """

        if type(sentences) is Sentence:
            sentences = [sentences]

        everything_embedded: bool = True

        if self.embedding_type == 'word-level':
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys(): everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)

        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """给句子中的每个词加入向量"""
        pass


class TokenEmbeddings(Embeddings):
    """所有词/字符级词向量的基础类"""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """返回词向量的长度"""
        pass

    @property
    def embedding_type(self) -> str:
        """返回 word-level"""
        return 'word-level'

class DocumentEmbeddings(Embeddings):
    """所有篇章级词向量的基础类"""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """返回词向量的长度"""
        pass

    @property
    def embedding_type(self) -> str:
        """返回 sentence-level"""
        return 'sentence-level'
