import torch
import torch.nn as nn
from torch.nn import functional as F
from Stable_diffusion.SD.attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    """Some Information about VAE_ResidualBlock"""
    # * 这个模块是由归一化层和卷积层组成的, 不改变size，依据输出的参数来改变卷积层filter的数量
    # * 在ResNet中很常用且常见的模块
    # * 只需要两个输入的参数: (in_channels, out_channels)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # * size不变
        
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # * size不变
        
        
        # * skip connection
        # * 之所以进行这个操作，是因为要确保残差相加时，残差的维度和output的维度保持一致，这样才能进行加法
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()  # * 定义了一个空操作 (noop operation)
        
        else:
            # * 注意kernel_size = 1
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)  # * size不变

    def forward(self, x):
        # * x: (batch, in_channel, height, width)
        
        residue = x
        
        x = self.groupnorm_1(x)
        
        x = F.silu(x) # * 激活函数一般放置在Normalization之后
        
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        
        x = F.silu(x)
        
        x = self.conv_2(x)
        
        # * output: (batch, out_channel, height, width)
        return x + self.residual_layer(residue)



'''
   # * Batch Normalization的每次计算是对每个channel的所有batch的所有像素点进行归一化
   # * Layer Normalization的每次计算是对每个图片的所有channel的所有像素点进行归一化
   # * Instance Normalization的每次计算是对每个图片的每个channel的所有像素点进行归一化
   # * Group Normalization的每次计算是对每个图片的每组channel的所有像素点进行归一化
   # * 归一化可以加速模型收敛, 防止loss function剧烈震荡
'''




class VAE_AttentionBlock(nn.Module):
    """Some Information about VAE_AttentionBlock"""
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        

    def forward(self, x):
        # * x: (batch, channels, height, width)
        residue = x
        
        b, c, h, w = x.shape
        
        # * 为attention输入做准备
        # * (batch, channels, height, width) -> (batch, channels, height * width) -> (batch, height * width, channels)
        x = x.view(b, c, h * w).transpose(-1, -2)
        
        # * (batch, height * width, channels) -> same
        x = self.attention(x)
        
        # * (batch, height * width, channels) -> (batch, height * width, channels) -> (batch, channels, height, width)
        x = x.transpose(-1, -2).view(b, c, h, w)
        
        x += residue

        return x




class VAE_Decoder(nn.Sequential):
    """Some Information about VAE_Decoder"""
    def __init__(self):
        super().__init__(
             
            
            
        )

    def forward(self, x):

        return x
