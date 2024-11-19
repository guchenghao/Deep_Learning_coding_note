import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """Some Information about SelfAttention"""
    # * hiddendim = channels
    def __init__(self, num_head, hiddendim, in_proj_bias=True, out_proj_bias=True, dropout=False):
        super().__init__()
        
        self.hiddendim = hiddendim
        self.num_head = num_head
        self.head_dim = hiddendim // num_head # * head_dim = hiddendim // num_head
        
        self.in_proj = nn.Linear(hiddendim, hiddendim * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(hiddendim, hiddendim, bias=out_proj_bias)
        self.dropout = dropout
        
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, x, attention_mask=False):
        
        batch, seq_len, _ = x.size()
        
        qkv_matrix = self.in_proj(x)
        
        # * 第二重境界
        q_matrix, k_matrix, v_matrix = torch.split(qkv_matrix, self.hiddendim, dim=-1)
        
        # * (batch, seq, hiddendim) -> (batch, seq, num_head, head_dim) -> (batch, num_head, seq, head_dim)
        q_multi_head = q_matrix.view(batch, seq_len, self.num_head, self.head_dim).transpose(2, 1)
        k_multi_head = k_matrix.view(batch, seq_len, self.num_head, self.head_dim).transpose(2, 1)
        v_multi_head = v_matrix.view(batch, seq_len, self.num_head, self.head_dim).transpose(2, 1)
        
        
        # * (batch, num_head, seq, seq)
        attention_matrix = q_multi_head @ k_multi_head.transpose(-1, -2) / math.sqrt(self.head_dim)
        
        
        if attention_mask:
            # * 返回上三角矩阵，由于是triu(1)，所以下三角和对角线均为0，上三角全为1，功能与tril()函数相反
            attention_mask = torch.ones_like(attention_matrix).triu(1)
            
            attention_matrix = attention_matrix.masked_fill(attention_mask == 0, float("-inf"))
        
        # * (batch, num_head, seq, seq)
        attention_weight = torch.softmax(attention_matrix, dim=-1)
        
        
        if self.dropout:
            attention_weight = self.dropout_layer(attention_weight)
        
        
        # * (batch, num_head, seq, seq) -> (batch, num_head, seq, head_dim)
        mid_output = attention_weight @ v_multi_head
        
        # * (batch, num_head, seq, head_dim) -> (batch, seq, num_head, head_dim)
        mid_output = mid_output.transpose(2, 1).contiguous()
        
        # * (batch, seq, num_head, head_dim) -> (batch, seq, hiddendim)
        mid_output = mid_output.view(batch, seq_len, -1)
        
        # * (batch, seq=(height * width), hiddendim=channels)
        output = self.out_proj(mid_output)
    

        return output