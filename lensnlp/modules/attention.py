import torch
import torch.nn.functional as F
import math
import copy


def dot_product_attention():
    pass


def addictive_attention():
    pass


def scaled_attention(query, key, value, mask=None, dropout=None):
    """ Scaled Attention

            Parameters
            ----------
            query : Tensor
                隐藏层尺寸
            key: TokenEmbeddings
                词向量
            value : Dictionary
                标签字典
            mask : str
                标签类型 如：'ner'
            dropout : bool
                是否用rnn
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
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = scaled_attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

