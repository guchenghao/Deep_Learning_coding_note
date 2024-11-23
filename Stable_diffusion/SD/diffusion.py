import torch 
import torch.nn as nn
from torch.nn import functional as F
from Stable_diffusion.attention import SelfAttention, CrossAttention


# * 一般需要进行堆叠的模块，要把Normalization放在最前面



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


class UpSample(nn.Module):
    """Some Information about UpSample"""
    def __init__(self, channels):
        super().__init__()
        
        self.conv_2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # * (batch, features, height, width) -> (batch, features,  height * 2, width * 2)
        
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # * 做最近邻插值和Upsample层的作用一样，都是通过插值的方式将图片的size放大
        
        x = self.conv_2d(x)
        

        return x




class UNET_OutputLayer(nn.Module):
    """Some Information about UNET_OutputLayer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # * x: (batch, 320, height / 8, width / 8)
        
        x = self.group_norm(x)
        
        x = F.silu(x)
        
        x = self.conv(x)
        
        
        # * (batch, 4, height / 8, width / 8)
        return x


class UNET_ResidualBlock(nn.Module):
    # * UNET的残差模块和VAE的残差模块的设计基本一致
    # ! UNET的残差模块多一个输入: (latent, time)
    """Some Information about UNET_ResidualBlock"""
    def __init__(self, in_channels, out_channels, time_dim = 1280):
        super().__init__()
        
        
        self.group_norm_features = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        
        self.group_norm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        
        self.linear_time = nn.Linear(time_dim, out_channels)  # * 对于时间步的嵌入向量只需要通过一个线性层进行处理
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
            
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        

    def forward(self, latent, time):
        # * latent: (batch, in_channels, height, width)
        # * time: (1, 1280)
        
        residue = latent
        
        latent = self.group_norm_features(latent)
        
        latent = F.silu(latent)
        
        latent = self.conv1(latent) # * (batch, in_channels, height, width) -> (batch, out_channels, height, width)
        
        
        # * 这里是与VAE的残差块不同的地方
        time = F.silu(time)
        
        time = self.linear_time(time) # * (1, 1280) -> (1, out_channels)
        
        # * 相当于将时间嵌入向量的每个channel的值，加到对应channel下，每个像素点上，让每个像素点都带上时间的信息再进行之后的卷积
        merged = latent + time.unsqueeze(-1).unsqueeze(-1) # * (batch, out_channels, height, width) + (1, out_channels, 1, 1) -> (batch, out_channels, height, width)
        
        merged = self.group_norm_merged(merged)
        
        merged = F.silu(merged)
        
        merged = self.conv_merged(merged)
        
        
        # * (batch, out_channels, height, width)
        # * 残差块的输出是结合了时间和潜在变量的信息
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    # * 这个AttentionBlock需要处理两个输入: 一个是结合了时间信息的latent; 一个是来自CLIP的文本嵌入向量
    # * 这个进行的是cross attention, 因为需要处理多模态的输入
    """Some Information about UNET_AttentionBlock"""
    def __init__(self, head_num, headdim, hiddendim_CLIP=768):
        super().__init__()
        channels = head_num * headdim
        
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layer_norm_1 = nn.LayerNorm(channels)
        self.self_attention = SelfAttention(head_num, channels, in_proj_bias=False)
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.cross_attention = CrossAttention(head_num, channels, in_proj_bias=False)
        self.layer_norm_3 = nn.LayerNorm(channels)
        
        # * GEGLU 激活函数（Gated Linear Unit with GeLU）是一种基于门控线性单元（GLU）和 Gaussian Error Linear Unit (GeLU) 激活函数结合的改进型激活函数。
        # * geglu = (xW1 + b1) * (xW2 + b2)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        

    def forward(self, latent_time, context_CLIP):
        # * latent_time: (batch, features, height, width)
        # * context_CLIP: (batch, seq, hiddendim=768)
        
        
        residue_long = latent_time  # * 最初的残差
        
        # * 先对latent_time进行处理
        latent_time = self.group_norm(latent_time)
        
        latent_time = self.conv_input(latent_time)
        
        b, c, h, w = latent_time.shape
        
        # * (batch, features, height, width) -> (batch, features, height * width)
        latent_time = latent_time.view(b, c, h*w).contiguous()
        
        # * (batch, features, height * width) -> (batch, height * width, features)
        latent_time = latent_time.transpose(-1, -2)
        
        
        # * Normalization + self-attention with skip connection
        # * latent_time: (batch, height * width, features)
        residue_short = latent_time  # * 做完selfAttention的残差
        
        latent_time = self.layer_norm_1(latent_time)
        latent_time = self.self_attention(latent_time)
        
        latent_time += residue_short
        
        
        # * Normalization + Cross-attention with skip connection
        residue_short = latent_time
        
        latent_time = self.layer_norm_2(latent_time)
        latent_time = self.cross_attention(latent_time, context_CLIP)
        
        latent_time += residue_short
        
        
        # * Normalization + FFN with GeGLU and skip connection
        residue_short = latent_time
        
        latent_time = self.layer_norm_3(latent_time)
        
        latent_time, gate = self.linear_geglu_1(latent_time).chunk(2, dim=-1)
        
        latent_time = latent_time * F.gelu(gate)
        
        
        latent_time = self.linear_geglu_2(latent_time)
        
        latent_time += residue_short
        
        # * (batch, height * width, features) -> (batch, features, height, width)
        latent_time = latent_time.transpose(-1, -2).view(b, c, h, w)
        
        
        return residue_long + self.conv_output(latent_time)







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



