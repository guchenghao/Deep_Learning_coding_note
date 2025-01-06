import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """Some Information about InputEmbeddings"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model  # * 嵌入向量的维度
        self.embedding = nn.Embedding(vocab_size, d_model)
        

    def forward(self, x):
        # * 这里的x表示的是索引值或者称为token
        # * shape of token is (batch, seq_len)
        
        # * shape is: (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PostionalEncoding(nn.Module):
    """Some Information about PostionalEncoding"""
    def __init__(self, d_model, seq_len, dropout_rate):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout_rate)

        # * 创建postionEncoding的嵌入矩阵, 用于与input_embedding进行相加
        # * shape is (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # * 构建位置信息向量postion
        # * shape is (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))

        # * 将sin函数应用位置向量的偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        # * 将cos函数应用位置向量的奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # * shape is (1, seq_len, d_model)

        # * 缓冲区是一种特殊类型的属性，主要用于存储不参与梯度计算的变量，但它们仍然是模型的一部分，并会在模型保存和加载时被处理。
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False)

        return self.dropout(x)  # * dropout层一般放置在计算层的后面


"""
   # * Channel指的也是特征维度
   # * Batch Normalization的每次计算是对每个channel的所有batch的所有像素点进行归一化
   # * Layer Normalization的每次计算是对每个图片的所有channel的所有像素点进行归一化
   # * Instance Normalization的每次计算是对每个图片的每个channel的所有像素点进行归一化
   # * Group Normalization的每次计算是对每个图片的每组channel的所有像素点进行归一化
   # * 归一化可以加速模型收敛, 防止loss function剧烈震荡
"""


# * 这里layernorm对每个句子分别在嵌入维度上进行归一化计算
# * 在layernorm中，一般会有两个参数gamma和beta，其中gamma是乘数，beta是加数，这两个参数是可学的参数
# * 这两个参数主要是给归一化的数据带来一些震荡
class LayerNormalization(nn.Module):
    """Some Information about LayerNormalization"""
    # * epsilon是为了防止分母过小，甚至为0
    def __init__(self, eps = 10 ** -6):
        super().__init__()
        
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # * Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # * Added
        
    def forward(self, x):
        
        mean = x.mean(dim=-1, keepdim=True)  # * 保留最后一个维度
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Some Information about FeedForwardBlock"""
    def __init__(self, d_model, d_ff, dropout):
        # * d_ff的大小是d_model的4倍
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        

    def forward(self, x):
        # * x shape is: (batch, seq_len, d_model)

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
    """Some Information about MultiHeadAttention"""
    def __init__(self, d_model, h, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.head_num = h
        assert d_model % h == 0

        self.head_dim = d_model // h
        self.dropout = nn.Dropout(dropout_rate)

        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)

        self.o_proj = nn.Linear(d_model, d_model)

    def attention(self, q_head, k_head, v_head, mask, dropout):

        # * (batch, head_num, seq_len, head_dim) -> (batch, head_num, seq_len, seq_len)
        attention_matrix = (q_head @ k_head.transpose(-1, -2)) / math.sqrt(q_head.shape[-1])

        if mask:
            attention_matrix = attention_matrix.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_matrix, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # * (barch, head_num, seq_len, head_dim)
        x = attention_scores @ v_head

        return x, attention_scores

    def forward(self, q, k, v, mask):

        batch, seq_len, _ = q.size()

        q_matrix = self.q_proj(q)
        k_matrix = self.k_proj(k)
        v_matrix = self.v_proj(v)

        q_head = q_matrix.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k_head = k_matrix.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        v_head = v_matrix.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(q_head, k_head, v_head, mask, self.dropout)

        # * (batch, seq_len, head_num, head_dim)
        x = x.transpose(1, 2).contiguous()

        x = x.view(batch, seq_len, -1)
        
        # * (batch, seq_len, d_model)
        x = self.o_proj(x)

        return x
