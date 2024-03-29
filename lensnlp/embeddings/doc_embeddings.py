from .base import DocumentEmbeddings, TokenEmbeddings
from .word_embeddings import StackedEmbeddings,FlairEmbeddings
import logging
from typing import List, Union

import torch
from lensnlp.models import nn
import torch.nn.functional as F

from lensnlp.utils.data import Token, Sentence
from lensnlp.hyper_parameters import Parameter,device

log = logging.getLogger('qhj')


class DocumentPoolEmbeddings(DocumentEmbeddings):

    """最简单的句向量计算方式

        Parameters
        ----------

        embeddings : List
            一个词向量列表，用于给句中的词加向量。
        mode : str
            三种pool的方式 ['mean', 'max', 'min']
        Examples
        --------
         >>> from lensnlp.Embeddings import WordEmbeddings, DocumentPoolEmbeddings
         >>> from lensnlp.utilis.data import Sentence
         >>> Embed_list = [WordEmbeddings('cn_glove')]
         >>> Docum_embed = DocumentPoolEmbeddings(Embed_list)
         >>> sent = Sentence('北京一览群智数据有限公司。')
         >>> Docum_embed.embed((sent)

        """

    def __init__(self, embeddings: List[TokenEmbeddings], mode: str = 'mean'):

        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.__embedding_length: int = self.embeddings.embedding_length

        self.to(device)

        self.mode = mode
        if self.mode == 'mean':
            self.pool_op = torch.mean
        elif mode == 'max':
            self.pool_op = torch.max
        elif mode == 'min':
            self.pool_op = torch.min
        else:
            raise ValueError(f'Pooling operation for {self.mode!r} is not defined')
        self.name: str = f'document_{self.mode}'

    @property
    def embedding_length(self) -> int:
        """返回向量的长度"""
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """
        :param sentences：单个Sentence或者Sentence的列表
        :return: 句子级别的向量，由词向量求得
        """
        everything_embedded: bool = True

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        for sentence in sentences:
            if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded:

            self.embeddings.embed(sentences)

            for sentence in sentences:
                word_embeddings = []
                for token in sentence.tokens:
                    token: Token = token
                    word_embeddings.append(token.get_embedding().unsqueeze(0))

                word_embeddings = torch.cat(word_embeddings, dim=0).to(device)

                if self.mode == 'mean':
                    pooled_embedding = self.pool_op(word_embeddings, 0)
                else:
                    pooled_embedding, _ = self.pool_op(word_embeddings, 0)

                sentence.set_embedding(self.name, pooled_embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocumentRNNEmbeddings(DocumentEmbeddings):

    """用RNN类网络为文本计算向量

        Parameters
        ----------

        embeddings : List
            一个词向量列表，用于给句中的词加向量。
        hidden_size : int
            输出的向量维度
        rnn_layers : int
            几层RNN
        reproject_words : bool
             是否线性转换一下词向量
        reproject_words_dimension : int
            如果线性转换词向量，输出的维度
        bidirectional : bool
            是否用双向RNN网络
        dropout : float
            dropout率
        locked_dropout : float
            locked_dropout率
        rnn_type : str
            'GRU', 'LSTM',  'RNN_TANH' or 'RNN_RELU'
        use_attention: bool
            是否使用注意力
        Examples
        --------

        """
    def __init__(self,
                 embeddings: List[TokenEmbeddings],
                 hidden_size=128,
                 rnn_layers=1,
                 reproject_words: bool = True,
                 reproject_words_dimension: int = None,
                 bidirectional: bool = False,
                 dropout_rate: float = 0.5,
                 word_dropout: float = 0.0,
                 locked_dropout: float = 0.0,
                 rnn_type='LSTM',
                 use_attention = False):

        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            if use_attention:
                self.__embedding_length *= 2
            else:
                self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(self.length_of_all_token_embeddings,
                                                     self.embeddings_dimension)

        self.rnn = torch.nn.LSTM(
            self.embeddings_dimension, hidden_size, num_layers=rnn_layers,bidirectional=self.bidirectional)

        self.use_attention = use_attention

        self.name = 'document_' + self.rnn._get_name()

        self.use_locked_dropout: bool = locked_dropout > 0.0
        if self.use_locked_dropout:
            self.l_dropout: torch.nn.Module = nn.LockedDropout(locked_dropout)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.use_word_dropout: bool = word_dropout > 0.0
        if self.use_word_dropout:
            self.word_dropout = nn.WordDropout(word_dropout)

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """
        :param sentences:
        :return: embedding存在了Sentence.embeddings中
        """
        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        longest_token_sequence_in_batch: int = len(sentences[0])
        all_sentence_tensors = []
        lengths: List[int] = []

        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.zeros(self.length_of_all_token_embeddings,
                                dtype=torch.float).unsqueeze(0)
                )

            word_embeddings_tensor = torch.cat(word_embeddings, 0).to(device)

            sentence_states = word_embeddings_tensor

            # 加到一个list中
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # 得到batch的特征
        sentence_tensor = torch.cat(all_sentence_tensors, 1)

        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.l_dropout(sentence_tensor)
        # 将变长序列进行padding之后压缩输入rnn模型中
        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

        self.rnn.flatten_parameters()

        rnn_out, hidden = self.rnn(packed)
        # rnn_out.shape = [seq,batch,hidden*num_direction]
        # hidden = (h_n, c_n) h_n.shape = [num_layer*num_direction, batch, hidden_size]
        # 将rnn输出进行解压缩，得到output和其实际句长
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.use_attention:
            from lensnlp.modules.attention import rnn_attention
            h_n, c_n = hidden
            # h_n.shape = [num_layes*direction, batch_size ,hidden_size]
            outputs = rnn_attention(outputs, h_n, self.rnn_layers)
            # outputs.shape = [batch_size, num_layers*direction*hidden_size]

        outputs = self.dropout(outputs)
        # use_attention获取句的特征
        if self.use_attention:
            for sentence_no in range(outputs.size()[0]):
                # print(sentence_no)
                embedding = outputs[sentence_no]
                sentences[sentence_no].set_embedding(self.name, embedding)

        # 获得句的特征
        else:
            for sentence_no, length in enumerate(lengths):
                last_rep = outputs[length - 1, sentence_no]

                embedding = last_rep
                if self.bidirectional:
                    first_rep = outputs[0, sentence_no]
                    embedding = torch.cat([first_rep, last_rep], 0)

                sentence = sentences[sentence_no]
                sentence.set_embedding(self.name, embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocumentCNN1DEmbedding(DocumentEmbeddings):

    """TextCNN 1D 来获得篇章级别的向量表示。

        Parameters
        ----------

        embeddings : List
            一个词向量列表，用于给句中的词加向量。
        embedding_type : str
            ‘static'：embedding不更新\'non-static'：embedding更新\'multichannel' ：双通道，只更新一个通道
        kernel_size : int
            卷积核的大小
        kernel_nums : bool
             卷积核个数
        in_channel : int
            当embedding_type 为双通道时，embedding更新的通道
        dropout : float
            dropout率
        Examples
        --------
         >>> from lensnlp.Embeddings import WordEmbeddings, DocumentCNN1DEmbedding
         >>> from lensnlp.utilis.data import Sentence
         >>> Embed_list = [WordEmbeddings('cn_glove')]
         >>> Docum_embed = DocumentCNN1DEmbedding(Embed_list)
         >>> sent = Sentence('北京一览群智数据有限公司。')
         >>> Docum_embed.embed((sent)

        """

    def __init__(self,
                 embeddings: List[TokenEmbeddings],
                 embedding_type: str = 'static',
                 dropout: float = 0.5,
                 kernel_size: List = [1,2,3,5],
                 kernel_nums: List = [256,256,256,256],
                 in_channel = 1,
                 hidden_size: int = 256
                 ):

        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.embedding_type = embedding_type
        self.in_channel = in_channel
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.kernel_nums = kernel_nums
        self.hidden_size = hidden_size

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length
        self.__embedding_length = hidden_size

        if self.embedding_type=='static':
            self.static_embeddings = True
        elif self.embedding_type=='non_static':
            self.static_embeddings = False
        elif self.embedding_type == 'multichannel':
            self.embeddings2: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
            self.embeddings2.weight.requires_grad = False
            self.in_channel = 2
        else:
            pass

        self.embeddings_dimension: int = self.length_of_all_token_embeddings

        self.convs = nn.ModuleList(
            [nn.Conv1d(self.in_channel, num, self.embeddings_dimension * size, stride=self.embeddings_dimension) for size, num in
             zip(self.kernel_size, self.kernel_nums)])
        self.fc = nn.Linear(sum(self.kernel_nums), self.hidden_size)

        self.to(device)

    def get_conv(self, i):
        return getattr(self, 'conv_%d' % i)


    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.convs.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        lengths: List[int] = []

        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.zeros(self.length_of_all_token_embeddings,
                                dtype=torch.float).unsqueeze(0)
                )

            word_embeddings_tensor = torch.cat(word_embeddings, 0).to(device)

            sentence_states = word_embeddings_tensor

            # 加到一个list中
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # 得到batch的特征
        sentence_tensor = torch.cat(all_sentence_tensors, 1)

        self.conv_results = [
            F.max_pool1d(F.relu(self.convs[i](sentence_tensor)), self.max_seq_len - self.kernel_sizes[i] + 1)
                .view(-1, self.kernel_nums[i])
            for i in range(len(self.convs))]
        x = torch.cat(self.conv_results, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        outputs = self.fc(x)

        # 获得句的特征
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[length - 1, sentence_no]

            embedding = last_rep

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocumentLMEmbeddings(DocumentEmbeddings):
    """利用预训练的字符级语言模型获得篇章级别的向量表示。

            Parameters
            ----------

            flair_embeddings : List
                一个flair词向量列表，用于给句中的词加向量。
            detach : bool
                是否更新flair模型中的权重
            Examples
            --------
             >>> from lensnlp.Embeddings import FlairEmbeddings, DocumentLMEmbeddings
             >>> from lensnlp.utilis.data import Sentence
             >>> Embed_list = [FlairEmbeddings('cn_forward_small'),FlairEmbeddings('cn_backward_small')]
             >>> Docum_embed = DocumentLMEmbeddings(Embed_list)
             >>> sent = Sentence('北京一览群智数据有限公司。')
             >>> Docum_embed.embed((sent)

            """
    def __init__(self, flair_embeddings: List[FlairEmbeddings], detach: bool = True):

        super().__init__()

        self.embeddings = flair_embeddings
        self.name = 'document_lm'

        self.static_embeddings = detach
        self.detach = detach

        self._embedding_length: int = sum(embedding.embedding_length for embedding in flair_embeddings)

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

            # iterate over sentences
            for sentence in sentences:

                # if its a forward LM, take last state
                if embedding.is_forward_lm:
                    sentence.set_embedding(embedding.name, sentence[-1]._embeddings[embedding.name])
                else:
                    sentence.set_embedding(embedding.name, sentence[0]._embeddings[embedding.name])

        return sentences


class DocumentCNN2DEmbedding(DocumentEmbeddings):
    """TextCNN 2D 来获得篇章级别的向量表示。

            Parameters
            ----------

            embeddings : List
                一个词向量列表，用于给句中的词加向量。
            embedding_type : str
                ‘static'：embedding不更新\'non-static'：embedding更新\'multichannel' ：双通道，只更新一个通道
            kernel_size : int
                卷积核的大小
            kernel_nums : bool
                 卷积核个数
            in_channel : int
                当embedding_type 为双通道时，embedding更新的通道
            dropout : float
                dropout率
            Examples
            --------
             >>> from lensnlp.Embeddings import WordEmbeddings, DocumentCNN2DEmbedding
             >>> from lensnlp.utilis.data import Sentence
             >>> Embed_list = [WordEmbeddings('cn_glove')]
             >>> Docum_embed = DocumentCNN2DEmbedding(Embed_list)
             >>> sent = Sentence('北京一览群智数据有限公司。')
             >>> Docum_embed.embed((sent)

            """

    def __init__(self,
                 embeddings: List[TokenEmbeddings],
                 embedding_type='static',
                 dropout: float = 0.5,
                 kernel_sizes: List = [3, 4, 5],
                 kernel_nums: List = [100, 100, 100],
                 in_channel=1,
                 cnn_type='CNNNKim',
                 hidden_size=128,
                 ):
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        # self.embeddings.requires_grad = False
        self.in_channel = in_channel
        self.dropout = dropout
        self.kernel_sizes = kernel_sizes
        self.kernel_nums = kernel_nums
        self.cnn_type = cnn_type
        self.hidden_size = hidden_size
        self.__embedding_length: int = hidden_size
        self.embedding_type_channel = embedding_type

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        assert (len(self.kernel_sizes) == len(self.kernel_nums))
        if self.embedding_type_channel == 'static':
            self.static_embeddings = True
        elif self.embedding_type_channel == 'non_static':
            self.static_embeddings = False
        elif self.embedding_type_channel == 'multichannel':
            self.embeddings2: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
            # self.embeddings2.requires_grad = True
            self.in_channel = 2
        else:
            pass

        self.embeddings_dimension: int = self.length_of_all_token_embeddings

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(self.in_channel, num, (size, self.embeddings_dimension)) for size, num in
             zip(self.kernel_sizes, self.kernel_nums)])
        # torch.nn.Conv2d() 输入的尺度是(N, C_in,H,W)，输出尺度（N,C_out,H_out,W_out）的计算方式：
        self.name = 'document_CNNKim'

        self.fc = torch.nn.Linear(sum(self.kernel_nums), self.hidden_size)

        self.to(device)

    def get_conv(self, i):
        return getattr(self, 'conv_%d' % i)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        if type(sentences) is Sentence:
            sentences = [sentences]

        self.convs.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        self.longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        lengths: List[int] = []

        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            for add in range(self.longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.zeros(self.length_of_all_token_embeddings,
                                dtype=torch.float).unsqueeze(0)
                )

            word_embeddings_tensor = torch.cat(word_embeddings, 0).to(device)

            sentence_states = word_embeddings_tensor  # shape是二维的，第一维是句长，每个batch不一致，第二维是token向量维度

            # 加到一个list中
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # 得到batch的特征
        sentence_tensor = torch.cat(all_sentence_tensors, 1)

        x_input = sentence_tensor.permute(1, 0, 2).unsqueeze(1)  # shape = [batch_size, in_channel, sen_len, embed_dim]

        if self.embedding_type_channel == "multichannel":
            sentence_tensor2 = sentence_tensor.permute(1, 0, 2).unsqueeze(1)
            x_input = torch.cat((x_input, sentence_tensor2), 1)  # (32,1,1000, 300)
        #    self.convs[i](x_input).size() = (32, 100, 998, 1)即（batch_size, out_channels, 卷积后长度，1）
        self.conv_results = [
            F.max_pool1d(F.relu(self.convs[i](x_input)).squeeze(3),
                         self.longest_token_sequence_in_batch - self.kernel_sizes[i] + 1)
                .view(-1, self.kernel_nums[i])
            for i in range(len(self.convs))]

        x = torch.cat(self.conv_results, 1)  # shape = [32,300]
        x = F.dropout(x, p=self.dropout)
        outputs = self.fc(x)  # shape = [batch_size, hidden_size]

        # 获得句的特征
        for sentence_no in range(outputs.size()[0]):
            embedding = outputs[sentence_no]
            sentences[sentence_no].set_embedding(self.name, embedding)
            # print(embedding.size())

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocmentRCNNEmbedding(DocumentEmbeddings):
    """RNN_CNN来获得篇章级别的向量表示。
    Parameters
    --------------------
    embeddings: List
        一个词向量列表，用于给句子中的词加向量
    stm_hidden_state: int
        rnn 隐藏节点数
    hidden_size: int
        cnn 后接的全连接层输出tensor最后一个维度size
    bidirectional: bool
        True:双向rnn, False:单向rnn
    dropout: float
        全连接层输出的dropout率
    word_dropout: float
        input dropout率
    locked_dropout: float
        全连接层输出的dropout
    >>> from lensnlp.Embeddings import WordEmbeddings, DocmentRCNNEmbedding
    >>> from lensnlp.utilis.data import Sentence
    >>> Embed_list = [WordEmbeddings('cn_glove')]
    >>> Docum_embed = DocmentRCNNEmbedding(Embed_list)
    >>> sent = Sentence('北京一览群智数据有限公司。')
    >>> Docum_embed.embed((sent)
    """
    def __init__(self,
                 embeddings: List[TokenEmbeddings],
                 lstm_hidden_state = 100,
                 hidden_size=128,
                 bidirectional: bool = False,
                 dropout: float = 0.5,
                 word_dropout: float = 0.0,
                 locked_dropout: float = 0.0, ):
        

        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.lstm_hidden_state = lstm_hidden_state
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length


        self.__embedding_length: int = self.hidden_size
        # if self.bidirectional:
        #     self.__embedding_length *= 2

        self.embeddings_dimension: int = self.length_of_all_token_embeddings

        self.lstm = torch.nn.LSTM(self.embeddings_dimension, self.lstm_hidden_state, bidirectional=self.bidirectional)
        # self.hidden = self.init_hidden(self.batch_size)
        self.content_dim = 256
        if self.bidirectional:
            self.conv = torch.nn.Conv1d(in_channels=self.lstm_hidden_state*2, out_channels=self.content_dim,
                                    kernel_size=self.lstm_hidden_state * 2, stride= self.embeddings_dimension)
        else:
            self.conv = torch.nn.Conv1d(in_channels=self.lstm_hidden_state, out_channels=self.content_dim,
                                        kernel_size=self.lstm_hidden_state * 2, stride=self.embeddings_dimension)
        self.fc = torch.nn.Linear(self.content_dim , self.hidden_size)

        self.name = 'document_' + self.lstm._get_name()

        if locked_dropout > 0.0:
            self.dropout: torch.nn.Module = nn.LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)

        self.use_word_dropout: bool = word_dropout > 0.0
        if self.use_word_dropout:
            self.word_dropout = nn.WordDropout(word_dropout)
        self.to(device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def init_hidden(self, batch_size):
        h0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.lstm_hidden_state).to(device))  # (1, 32, 128)
        c0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.lstm_hidden_state).to(device))  # (1, 32, 128)
        return (h0, c0)

    def embed(self, sentences: Union[List[Sentence], Sentence]):

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.lstm.zero_grad()
        self.conv.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        lengths: List[int] = []

        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.zeros(self.length_of_all_token_embeddings,
                                dtype=torch.float).unsqueeze(0)
                )

            word_embeddings_tensor = torch.cat(word_embeddings, 0).to(device)

            sentence_states = word_embeddings_tensor  # shape = [seq_len, emb_dim]

            # 加到一个list中
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # 得到batch的特征
        sentence_tensor = torch.cat(all_sentence_tensors, 1)  # shape = [seq_len, batch_size, emb_dim]
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        sentence_tensor = self.dropout(sentence_tensor)
        # print(sentence_tensor.size())
        start_hidden = self.init_hidden(sentence_tensor.size()[1])

        lstm_out, hidden = self.lstm(sentence_tensor, start_hidden)  # rnn_out = [seq_len, batch_size, hidden_size]

        conv_out = self.conv(lstm_out.permute(1, 2, 0))  # conv_out = [batch_size, content_dim, seq_len - ken_size + 1/stride]
        outputs = self.fc(conv_out.squeeze(2))
        outputs = self.dropout(outputs)

        # 获得句的特征
        for sentence_no in range(outputs.size()[0]):
            embedding = outputs[sentence_no]
            sentences[sentence_no].set_embedding(self.name, embedding)
            # print(embedding.size())

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass
