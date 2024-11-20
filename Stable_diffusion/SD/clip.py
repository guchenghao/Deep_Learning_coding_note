import torch
import torch.nn as nn
from torch.nn import functional as F
from Stable_diffusion.attention import SelfAttention

class CLIPEmbedding(nn.Module):
    """Some Information about CLIPEmbedding"""
    def __init__(self, n_vocab, hiddendim, n_tokens):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, hiddendim)
        
        # * 这里位置编码并没有使用transformer中常用的正弦和余弦函数来进行位置编码，而是将位置编码设置为可学习的参数，让模型来进行学习
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, hiddendim))
        

    def forward(self, tokens):
        # * tokens: (batch, seq)
        
        
        # * (batch, seq) -> (batch, seq, hiddendim)
        tokens = self.token_embedding(tokens)
        
        
        tokens += self.position_embedding  # * 添加位置编码
        
        return tokens




class CLIPLayer(nn.Module):
    """Some Information about CLIPLayer"""
    def __init__(self, num_head, hiddendim):
        super().__init__()
        
        self.attention = SelfAttention(num_head, hiddendim)
        self.layer_norm_att = nn.LayerNorm(hiddendim)
        self.layer_norm_ffn = nn.LayerNorm(hiddendim)
        
        self.up_dim = nn.Linear(hiddendim, hiddendim * 4)
        self.down_dim = nn.Linear(hiddendim * 4, hiddendim)
        

    def forward(self, x):
        # * x: (batch, seq, hiddendim)
        
        
        # ! Multi-head Attention
        residue = x
        
        x = self.layer_norm_att(x) # * 先做layer normalization
        
        x = self.attention(x, True) # * 将mask设置为True
        
        x += residue
        
        
        # ! FFN
        residue = x
        
        x = self.layer_norm_ffn(x) # * 先做layer normalization
        
        x = self.up_dim(x)
        
        x = F.glu(x) # * QucikGElLU, 因为用这个函数效果好
        
        x = self.down_dim(x)
        
        x += residue
        
        # * (batch, seq, hiddendim)
        return x












# * clip的结构: 实际上就是多个transfomer的encoder的堆叠
class CLIP(nn.Module):
    """Some Information about CLIP"""
    def __init__(self):
        super().__init__()
        
        # * 49408是CLIP的词汇表的总数，786是嵌入向量的维度(hiddendim)，77是sequence的最大长度 (token的最大数量)
        self.embedding = CLIPEmbedding(49408, 768, 77) 
        
        # * 12是head的数量，768是hidden_dim
        # * 堆叠了12个
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])
        
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens):
        
        tokens = tokens.type(torch.long)  # * 用longTensor的类型
        
        # * (batch, seq) -> (batch, seq, hiddendim)
        state = self.embedding(tokens)
        
        
        for layer in self.layers:
            state = layer(state)
        
        # * (batch, seq, hiddendim)
        output = self.layer_norm(state)

        return output