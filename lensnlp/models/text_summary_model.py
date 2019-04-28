from . import nn
from typing import Union, List
from lensnlp.utilis.data import Dictionary, Sentence, Token, Label
import torch


class TextSummary(nn.Model):
    def __init__(self):
        super(TextSummary, self).__init__()
        pass

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        pass

    def predict(self, sentences: Union[List[Sentence], Sentence], mini_batch_size=32):
        pass

    def forward_labels_and_loss(self, sentences: Union[List[Sentence], Sentence]) -> (List[List[Label]], torch.tensor):
        pass

