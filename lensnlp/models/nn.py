import torch.nn

from abc import abstractmethod

from typing import Union, List

from lensnlp.utilis.data import Sentence, Label


class Model(torch.nn.Module):

    @abstractmethod
    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        pass

    @abstractmethod
    def forward_labels_and_loss(self, sentences: Union[List[Sentence], Sentence]) -> (List[List[Label]], torch.tensor):
        pass

    @abstractmethod
    def predict(self, sentences: Union[List[Sentence], Sentence], mini_batch_size=32) -> List[Sentence]:
        pass


class LockedDropout(torch.nn.Module):
    """
    随机扔embedding中的值
    """
    def __init__(self, dropout_rate=0.5):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x


class WordDropout(torch.nn.Module):
    """
    随机扔embedding
    """
    def __init__(self, dropout_rate=0.05):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), 1, 1).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x
