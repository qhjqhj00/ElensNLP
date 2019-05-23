import torch
import torch.nn.functional as F
import math
import copy


def dot_product_attention():
    pass


def addictive_attention():
    pass


def rnn_attention(rnn_out, hidden_state, rnn_layers):
    """
    简单的dot product attention
    :param rnn_out: rnn的输出 rnn_out.shape = [seq,batch,hidden_size*num_direction]
    :param hidden_state: 序列的隐藏状态 hidden = (h_n, c_n) h_n.shape = [num_layer*num_direction, batch, hidden_size]
    :param rnn_layers: RNN类网络的层数
    :return: 每个cell的一个注意力权重
    """
    batch_size = hidden_state.size()[1]
    state = hidden_state.view(rnn_layers, batch_size, -1)
    merged_state = torch.cat([s for s in state],1)  # shape = [batch, hidden*direction]
    merged_state = merged_state.unsqueeze(2)
    # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
    weights = torch.bmm(rnn_out.permute(1, 0, 2), merged_state)
    weights = torch.nn.functional.softmax(weights.squeeze(2),dim=1).unsqueeze(2)
    # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
    return torch.bmm(rnn_out.permute(1, 2, 0), weights).squeeze(2)


def clones(module, N):
    """产生相同的module"""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(torch.nn.Module):
    """ Multihead Attention
            Parameters
            ----------
            h : int
            头的数量
            d_model: int
            模型size
            dropout : float
            dropout率
            Examples
            --------
             >>> from lensnlp.modules.attention import MultiHeadedAttention
             >>> MHA = MultiHeadedAttention(8,512)
            """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 完成所以的线性转换 d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) 完成所以的scaled dot attention
        x, self.attn = self.scaled_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 将attention的输出连接，再通过全连结层
        x = x.transpose(1, 2).contiguous() .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    @staticmethod
    def scaled_attention(query, key, value, mask=None, dropout: float = None):
        """ Scaled Attention

                Parameters
                ----------
                query : Tensor
                key: Tensor
                value : Tensor
                mask : bool
                dropout : torch.nn.Dropout
                """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = torch.nn.Dropout(p_attn,p=dropout)
        return torch.matmul(p_attn, value), p_attn
