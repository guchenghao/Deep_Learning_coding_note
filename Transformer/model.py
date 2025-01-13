import torch
import torch.nn as nn
import math


# * 从vocabulary映射到Word Vector (Word Embedding)
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
        
        # * 广播机制
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

# * 构建transformer中skip connection功能
class ResidualConnection(nn.Module):
    """Some Information about ResidualConnection"""
    def __init__(self, dropout_rate):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.norm = LayerNormalization()
        

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""
    def __init__(self, self_attention_block, feed_forward_block, dropout_rate):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout_rate)
        
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(2)])

    def forward(self, x, src_mask):
        # * src_mask在encoder的过程中，可以进行padding mask，保证序列长度相同，以及控制注意力的范围
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    """Some Information about Encoder"""
    def __init__(self, layers):
        # * layers由一组EncoderBlock组成
        super().__init__()
        
        self.layers = layers
        
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    """Some Information about DecoderBlock"""
    def __init__(self, self_attention_block, cross_attention_block, feed_forward_block, dropout_rate):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(3)])
        

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # * src_mask作用于encoder，tgt_mask作用于decoder
        # * 为什么需要这两种mask? 是因为在类似于机器翻译这种任务中源语言与目标语言不一样，需要设计不同的mask
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block)
        
        return x


class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        for layer in self.layers:
            
            x = layer(x, encoder_output, src_mask, tgt_mask)
        

        return self.norm(x)


# * 将Decoder的输出从词向量空间映射到vocab空间
class ProjectionLayer(nn.Module):
    """Some Information about ProjectionLayer"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
        

    def forward(self, x):
        # * x: (batch, seq_len, d_model)
        
        # * (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        # * 这里使用log_softmax是为了在训练阶段，数值更加稳定且高效
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """Some Information about Transformer"""
    def __init__(self, encoder, decoder, src_embedding, tgt_embedding, src_position, tgt_position, projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_position(src)

        return self.encoder(src, src_mask)


    def decode(self, encoder_output, src_mask, tgt_mask, tgt):
        tgt = self.tgt_embedding(tgt)
        tgt = self.src_position(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    
    def projection(self, decoder_output):
        
        return self.projection_layer(decoder_output)


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, h=8, dropout_rate=0.1, d_ff=2048):

    # * embedding layer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # * positional encoding layers
    src_pos = PostionalEncoding(d_model, src_seq_len, dropout_rate)
    tgt_pos = PostionalEncoding(d_model, tgt_seq_len, dropout_rate)


    # * encoder block
    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout_rate)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout_rate)
        encoder_blocks.append(encoder_block)
    
    
    # * decoder block
    decoder_blocks = []
    
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout_rate)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout_rate)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)
        
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout_rate)
        decoder_blocks.append(decoder_block)
    
    
    # * encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    
    # * projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # * Initial the parameter
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    
    return transformer
    
