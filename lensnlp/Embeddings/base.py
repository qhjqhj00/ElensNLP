from abc import abstractmethod
from typing import List, Union
import torch

from lensnlp.utilis.data import Sentence


class Embeddings(torch.nn.Module):
    """所有词向量类的基础类"""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """返回向量的长度"""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """给词加上词向量"""

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
        """Private method for adding embeddings to all words in a list of sentences."""
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
        return 'word-level'

class DocumentEmbeddings(Embeddings):
    """所有篇章级词向量的基础类"""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        pass

    @property
    def embedding_type(self) -> str:
        return 'sentence-level'
