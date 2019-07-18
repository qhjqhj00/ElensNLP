import warnings
import logging
from pathlib import Path

from . import nn
import torch
from typing import List, Union
from lensnlp.embeddings import *
from lensnlp.utils.data import Dictionary, Sentence, Label, Token
from lensnlp.hyper_parameters import Parameter,device
from lensnlp.utils.training_utils import convert_labels_to_one_hot, clear_embeddings
import os

import random

import torch.nn.functional as F
from lensnlp.modules import attention

log = logging.getLogger('lensnlp')

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(input_dim, emb_dim)

        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(output_dim, emb_dim)

        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = torch.nn.Linear(hid_dim, output_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class RNN2RNN(torch.nn.Module):

    def __init__(self,
                 embeddings: WordEmbeddings,
                 input_dim: int,
                 output_dim: int,
                 src_dict,
                 trg_dict,
                 encoder_dim: int = 256,
                 decoder_dim: int = 256,
                 hidden_size: int = 512,
                 n_layers: int = 2,
                 encoder_dropout: float = 0.5,
                 decoder_dropout: float = 0.5
                 ):

        super(RNN2RNN,self).__init__()

        self.src_dict = src_dict
        self.trg_dict = trg_dict

        PAD_IDX = self.src_dict['pad']

        self.embeddings = embeddings
        self.encoder = Encoder(input_dim, encoder_dim, hidden_size, n_layers, encoder_dropout)
        self.decoder = Decoder(output_dim, decoder_dim, hidden_size, n_layers, decoder_dropout)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index = PAD_IDX)

        self.start_embed = 0 #TODO

        assert self.encoder.hidden_size == self.decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self.apply(self.init_weights)

        self.to(device)

    @staticmethod
    def init_weights(m):
        for name, param in m.named_parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src, teacher_forcing_ratio, trg = None):

        batch_size = src.shape[1]
        if trg is not None:
            max_len = trg.shape[0]
        else:
            max_len = src.shape[0] + 5
            trg = self.start_embed
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        outputs = outputs[1:].view(-1, outputs.shape[-1])

        return outputs

    def forward_loss(self, src, trg, teacher_forcing_ratio):
        outputs = self.forward(src, trg, teacher_forcing_ratio)
        loss = self.loss_function(outputs, trg)
        return loss

    def save(self, model_file: Union[str, Path]):
        """
        存模型
        """
        model_state = {
            'state_dict': self.state_dict(),
            'src_dict': self.src_dict,
            'trg_dict': self.trg_dict,
            'embeddings': self.embeddings}
        torch.save(model_state, str(model_file), pickle_protocol=4)

    def predict(self, src, mini_batch_size: int = 32):
        """
        预测
        输入为 Sentence 数量不限
        返回 Sentence，标签存入对应的位置
        mini_batch_size为每个batch预测的数量
        """
        with torch.no_grad():
            batches = [filtered_sentences[x:x + mini_batch_size] for x in range(0, len(filtered_sentences), mini_batch_size)]

            for batch in batches:
                outputs = self.forward(batch, teacher_forcing_ratio=0)
                predicted_seq = torch.argmax(outputs, dim=1)
                # TODO
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

        model = RelationExtraction(
            embeddings=state['relation_embeddings'],
            label_dictionary=state['label_dictionary'],
        )
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(device)

        return model

    @classmethod
    def _load_state(cls, model_file: Union[str, Path]):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            state = torch.load(str(model_file), map_location=device)
            return state

    def load(model_file: str):
        if model_file == 'seq2seq':
            classifier: RelationExtraction = RelationExtraction.load_from_file(Path(CACHE_ROOT) / 'seq2seq/best-mdoel.pt')
        else:
            try:
                classifier: RelationExtraction = RelationExtraction.load_from_file(Path(CACHE_ROOT) / model_file)
            except NameError('specify a model!'):
                raise
        return classifier