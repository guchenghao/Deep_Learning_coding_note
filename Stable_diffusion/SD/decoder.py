import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention


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
        
        x = self.groupnorm_1(x) # * 为了完成block的堆叠，先做normalization，可以理解为是对上一个block的输入先进行归一化
        
        x = F.silu(x) # * 激活函数一般放置在Normalization之后
        
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        
        x = F.silu(x)
        
        x = self.conv_2(x)
        
        # * output: (batch, out_channel, height, width)
        return x + self.residual_layer(residue)



'''
   # * 归一化层的在CV中的作用: 
   # * Batch Normalization的每次计算是对每个channel的所有batch的所有像素点进行归一化
   # * Layer Normalization的每次计算是对每个图片的所有channel的所有像素点进行归一化
   # * Instance Normalization的每次计算是对每个图片的每个channel的所有像素点进行归一化
   # * Group Normalization的每次计算是对每个图片的每组channel的所有像素点进行归一化
   # * 归一化可以加速模型收敛, 防止loss function剧烈震荡
   
   # ! Layer normalization在NLP领域中的使用与CV中有所不同，而且Layer norm在CV中也不常用
   # ! 在NLP中，Layer norm是对每个batch的每个token进行归一化
   # ! 在CV中，Layer norm是对每个图片的所有channel进行归一化
'''




class VAE_AttentionBlock(nn.Module):
    """Some Information about VAE_AttentionBlock"""
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels) # * head默认为1
        

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
        # *
        super().__init__(
            
            
             # ! 输入阶段
             # * (batch, 4, height / 8, width / 8) -> same
             nn.Conv2d(4, 4, kernel_size=1),
             
             
             # ! 残差卷积和自注意力阶段
             # * 这个过程是到着encoder的顺序，进行设计的
             # * (batch, 4, height / 8, width / 8) -> (batch, 512, height / 8, width / 8), size不变
             # * 提升channel的数量
             nn.Conv2d(4, 512, kernel_size=3, padding=1),
             
             VAE_ResidualBlock(512, 512),
             
             VAE_AttentionBlock(512),
             
             VAE_ResidualBlock(512, 512),
             
             VAE_ResidualBlock(512, 512),
             
             VAE_ResidualBlock(512, 512),
             
             # * (batch, 512, height / 8, width / 8) -> same
             VAE_ResidualBlock(512, 512),
             
             
             # ! 上采样1
             # * (batch, 512, height / 8, width / 8) -> (batch, 512, height / 4, width / 4)
             # * 一般来说，upsample会结合conv2d一起使用，upsampling通过插值的方式还原图像的size，conv2d来补充采样细节
             nn.Upsample(scale_factor=2),
             
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             
             VAE_ResidualBlock(512, 512),
             VAE_ResidualBlock(512, 512),
             # * (batch, 512, height / 4, width / 4)
             VAE_ResidualBlock(512, 512),
             
             
             # ! 上采样2
             # * (batch, 512, height / 4, width / 4) -> (batch, 512, height / 2, width / 2)
             nn.Upsample(scale_factor=2),
             
             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             
             VAE_ResidualBlock(512, 256),
             VAE_ResidualBlock(256, 256),
             # * (batch, 256, height / 2, width / 2)
             VAE_ResidualBlock(256, 256),
             
             
             # ! 上采样3
             # * (batch, 256, height / 2, width / 2) -> (batch, 256, height, width)
             nn.Upsample(scale_factor=2),
             
             nn.Conv2d(256, 256, kernel_size=3, padding=1),
             
             VAE_ResidualBlock(256, 128),
             VAE_ResidualBlock(128, 128),
             # * (batch, 128, height, width)
             VAE_ResidualBlock(128, 128),
             
             
             # ! 输出阶段
             nn.GroupNorm(32, 128),
             
             nn.SiLU(),
             
             # * (batch, 128, height, width) -> (batch, 3, height, width)
             nn.Conv2d(128, 3, kernel_size=3, padding=1)
             
        )

    def forward(self, x):
        # * x: (batch, 4, height / 8, height / 8)
        
        x /= 0.18215
        
        # * 顺序执行decoder中每个层
        for module in self:
            x = module(x)
        
        
        # * (batch, 3, height, width)
        return x
