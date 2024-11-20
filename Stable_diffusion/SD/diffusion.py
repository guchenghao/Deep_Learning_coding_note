import torch 
import torch.nn as nn
from torch.nn import functional as F
from Stable_diffusion.attention import SelfAttention, CrossAttention


# * 由两个linear层组成, 中间加个激活函数
class TimeEmbedding(nn.Module):
    """Some Information about TimeEmbedding"""
    def __init__(self, n_embed):
        super().__init__()
        self.up_dim = nn.Linear(n_embed, n_embed * 4)
        self.output_dim = nn.Linear(n_embed * 4, n_embed * 4)
        

    def forward(self, x):
        # * x: (1, 320)
        
        x = self.up_dim(x)
        
        x = F.silu(x)
        
        x = self.output_dim(x)
        
        
        # * (1, 1280)
        return x


# * 这个方法因为不需要初始化，所以可以不写初始化方法
class SwitchSequential(nn.Sequential):
    """Some Information about SwitchSequtial"""

    def forward(self, latent, context_CLIP, time):
        for layer in self:
            # * isinstance(): 用于判断一个对象是否是某种特定类型或类的实例
            if isinstance(layer, UNET_AttentionBlock):
                latent = layer(latent, context_CLIP) # * 计算交叉注意力
            
            elif isinstance(layer, UNET_ResidualBlock):
                latent = layer(latent, time) # * 残差模块只接受时间步和潜在向量作为输入
            
            else:
                latent = layer(latent) # * 如果只是卷积层的话，只接收latent作为输入
                

        return latent










class UNET(nn.Module):
    """Some Information about UNET"""
    def __init__(self):
        super().__init__()
        
        # * 减少size且增加features的数量
        self.encoders = nn.ModuleList([
            
            
            # ! 输入阶段
            # * (batch, 4, height / 8, width / 8) -> (batch, 320, height / 8, width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8 ,40)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8 ,40)),
            
            
            # ! 下采样1
            # * (2 - 3) / 2 + 1; 对于奇数HW的图像来说size取半，且向上取整；对于偶数HW的图像来说，size取半
            # * (batch, 320, height / 8, width / 8) -> (batch, 320, height / 16, width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8 ,80)),
            
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8 ,80)),
            
            
            # ! 下采样2
            # * (batch, 640, height / 16, width / 16) -> (batch, 640, height / 32, width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8 ,160)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8 ,160)),
            
            
            # ! 下采样3
            # * (batch, 1280, height / 32, width / 32) -> (batch, 1280, height / 64, width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # * (batch, 1280, height / 64, width / 64) -> same
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])
        
        
        self.bottleneck = SwitchSequential([
            UNET_ResidualBlock(1280, 1280),
            
            UNET_AttentionBlock(8 ,160),
            
            UNET_ResidualBlock(1280, 1280)
        ])
        
        
        
        self.decoder = nn.ModuleList([
            
            # * (batch, 2560, height / 64, width / 64) -> (batch, 1280, height / 64, width / 64)
            # ! 因为需要考虑到skip connection，所以这里的channel的数量翻倍为2560
            # ! 输入阶段
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            
            # ! 上采样1
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8 ,160)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8 ,160)),
            
            
            # ! 上采样2
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8 ,160), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8 ,80)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8 ,80)),
            
            
            # ! 上采样3
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8 ,80), UpSample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8 ,40)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8 ,40)),
            
            
            # ! 输出前处理
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8 ,40))
            
        ])

    def forward(self, x):

        return x








# * 这是扩散模型的去噪过程
class Diffusion(nn.Module):
    """Some Information about Diffusion"""
    def __init__(self):
        super().__init__()
        # * 将每次输入的时间步向量也转化为嵌入向量，输入到Unet中
        # * 每次输入对应一个唯一的时间步
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        
        self.final = UNET_Outputlayer(320, 4)
        

    def forward(self, latent, context_CLIP, time):
        # * latent: (batch, 4, height / 8, width / 8)
        # * context_CLIP: (batch, seq, hiddendim=768)
        # * time: (1, 320)
        
        
        # * 这个主要是为了让模型知道现在处于去噪过程中哪一步
        # * (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # * (batch, 4, height / 8, width / 8) -> (batch, 320, height / 8, width / 8)
        output = self.unet(latent, context_CLIP, time)
        
        
        # *  (batch, 320, height / 8, width / 8) -> (batch, 4, height / 8, width / 8)
        output = self.final(output)
        

        return output



