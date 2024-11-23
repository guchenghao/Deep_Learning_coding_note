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





class CrossAttention(nn.Module):
    """Some Information about CrossAttention"""
    def __init__(self, head_num, hiddendim_ky, hiddendim_q, in_proj_bias=True, out_proj_bias=True, dropout=False):
        super().__init__()
        
        # * 相当于以query的hidden dimention为基准
        self.q_proj = nn.Linear(hiddendim_q, hiddendim_q, bias=in_proj_bias)
        
        self.k_proj = nn.Linear(hiddendim_ky, hiddendim_q, bias=in_proj_bias)
        self.v_proj = nn.Linear(hiddendim_ky, hiddendim_q, bias=in_proj_bias)
        self.out_proj = nn.Linear(hiddendim_q, hiddendim_q, bias=out_proj_bias)
        
        
        self.head_num = head_num
        self.head_dim = hiddendim_q // head_num
        
        
        self.dropout_layer = nn.Dropout(0.1)
        self.dropout = dropout
        
        
        
    def forward(self, x, y):
        # * x is query (latent): (batch, seq_len_q, dim_q)
        # * y is key, value (context): (batch, seq_len_kv, dim_kv) = (batch, 77, 768)
        # * 没有mask
        
        # * b1 == b2
        b1, seq_q, _ = x.shape
        b2, seq_kv, _ = y.shape
        
        # * (batch, seq_len_q, dim_q) -> same
        query_matrix = self.q_proj(x)
        
        # * (batch, seq_len_kv, dim_kv) -> (batch, seq_len_kv, dim_q)
        key_matrix = self.k_proj(x)
        
        # * (batch, seq_len_kv, dim_kv) -> (batch, seq_len_kv, dim_q)
        value_matrix = self.v_proj(x)
        
        # * (batch, seq_len_q, dim_q) -> (batch, head_num, seq_len_q, head_dim)
        query_multi_head = query_matrix.view(b1, seq_q, self.head_num , self.head_dim).transpose(2, 1)
        # * (batch, seq_len_kv, dim_q) -> (batch, head_num, seq_len_kv, head_dim)
        key_multi_head = key_matrix.view(b2, seq_kv, self.head_num , self.head_dim).transpose(2, 1)
        # * (batch, seq_len_kv, dim_q) -> (batch, head_num, seq_len_kv, head_dim)
        value_multi_head = value_matrix.view(b2, seq_kv, self.head_num , self.head_dim).transpose(2, 1)
        
        # * (batch, head_num, seq_len_q, seq_len_kv)
        attention_matrix = query_multi_head @ key_multi_head.transpose(-1, -2) / math.sqrt(self.head_dim)
        
        
        attention_weights = torch.softmax(attention_matrix, dim=-1)
        
        
        if self.dropout:
            attention_weights = self.dropout_layer(attention_weights)
            
        
        # * (batch, head_num, seq_len_q, head_dim)
        mid_output = attention_weights @ value_multi_head
        
        # * (batch, head_num, seq_len_q, head_dim) -> (batch, seq_len_q, hiddendim_q)
        mid_output = mid_output.transpose(1, 2).contiguous()
        mid_output = mid_output.view(b1, seq_q, -1)
        
        output = self.out_proj(mid_output)
        
        
        # * (batch, seq_len_q, hiddendim_q)
        return output