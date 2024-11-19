import torch
import torch.nn as nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock




# * 为什么要这么去设计Encoder: 实际上在deep learning开发社区自己造轮子时，很多情况下，我们会直接搬运之前效果比较好的模型的结构，没有具体原因，很多时候就是因为效果好
# * 可能对SD开发团队来说，他们也去直接搬运了之前一些研究或者开发团队的encoder的设计框架，单纯是因为效果好
# * 如果只是简单地添加一些新的功能（如添加初始化方法），但不修改 forward 流程，可以继承 nn.Sequential。
# * 如果需要自定义 forward 方法以实现更复杂的逻辑，应该继承自 nn.Module。
class VAE_Encoder(nn.Sequential):
    """Some Information about VAE_Encoder"""
    def __init__(self):
        # * Encoder的总体思路就是减小图像的size的同时，增加channel的数量
        # * 输入阶段 -> 降维1 -> 降维2 -> 降维3 -> Self-Attention -> Group Normalization -> 输出阶段
        super().__init__(
            
            # ! 输入阶段
            # * (batch, channel=3, height, width) -> (batch, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1), # * size不变
            
            # * 这个残差块和常用的残差块模块基本相同，由卷积和BN层组成
            # * 残差模块不会改变图像的size
            # * (batch, 128, height, width) -> same
            VAE_ResidualBlock(128, 128),
            
            # * (batch, 128, height, width) -> same
            VAE_ResidualBlock(128, 128),
            
            
            
            # ! 降维1
            # * (batch, 128, height, width) -> (batch, 128, height / 2, width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2), # * size减半，降维
            
            # * (batch, 128, height / 2, width/ 2) -> (batch, 256, height/ 2, width/ 2)
            VAE_ResidualBlock(128, 256),
            
            # * (batch, 256, height/ 2, width/ 2) -> same
            VAE_ResidualBlock(256, 256),
            
            
            
            # ! 降维2
            # * (batch, 256, height/ 2, width/ 2) -> (batch, 256, height/ 4, width/ 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2), # * size减半，降维
            
            # * (batch, 256, height / 4, width/ 4) -> (batch, 512, height/ 4, width/ 4)
            VAE_ResidualBlock(256, 512),
            
            # * (batch, 512, height/ 4, width/ 4) -> same
            VAE_ResidualBlock(512, 512),
            
            
            
            # ! 降维3
            # * (batch, 512, height/ 4, width/ 4) -> (batch, 512, height/ 8, width/ 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2), # * size减半，降维
            
            # * 这3个残差block不改变channel的数量和图像的size大小
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            # * (batch, 512, height/ 8, width/ 8) -> same
            VAE_ResidualBlock(512, 512),
            
            
            
            # ! Self-attention
            # * 对像素点进行Self-attention
            # * (batch, 512, height/ 8, width/ 8) -> (batch, 512, height/ 8, width/ 8)
            VAE_AttentionBlock(512),
            
            # * (batch, 512, height/ 8, width/ 8) -> same
            VAE_ResidualBlock(512, 512),
            
            
            
            # ! 归一化
            # * GN 通常用于图像任务（如分类、目标检测），尤其是在小批量或小数据集的场景中
            # * Batch Normalization 的效果依赖于较大的批量大小（通常需要batch size > 32）。当批量较小时，BN 的统计特性会显著波动，导致模型性能下降。
            # * (batch, 512, height/ 8, width/ 8) -> same
            # * 一般来说，归一化层不会改变size
            nn.GroupNorm(32, 512),
            
            # * SiLU函数来自于sigmoid函数，选择这个激活函数的原因单纯是因为这个激活函数效果要更好，没有别的特殊原因
            nn.SiLU(),
            
            
            
            # ! 输出阶段
            # * Bottleneck
            # * (batch, 512, height/ 8, width/ 8) -> (batch, 8, height/ 8, width/ 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1), # * 不改变size的大小
            
            # * (batch, 8, height/ 8, width/ 8) -> (batch, 8, height/ 8, width/ 8)
            nn.Conv2d(8, 8, kernel_size=1), # * 不改变size的大小
        )
        

    def forward(self, x, noise):
        # *  x: (batch, Channel, height, width)
        # *  noise: (batch, Output_Channel, height/ 8, width/ 8)
        
        
        
        # ! 顺序执行所有的初始化过的层
        # * 这些层是可以复用，或者迁移学习的
        for module in self:
            # * 找到所有的stride=2的卷积层
            if getattr(module, "stride", None) == (2, 2):
                
                # * (Padding_left, Padding_right, Padding_top, Padding_bottom)
                # * 对x做不对称的padding填充，默认填充0
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        
        
        # ! Variational process (Reparameter Trick)
        # * 将输出分割为两个tensor: 一个表示mean；一个表示log_variance
        # * (batch, 8, height / 8, width / 8) -> two tensors of shape: (batch, 4, height / 8, width / 8)
        mean, log_var = torch.chunk(x, 2, dim=1)
        
        # * 控制log_variance的取值范围，不要过大过小
        # *  (batch, 4, height / 8, width / 8) -> same
        log_var = torch.clamp(log_var, -30, 20)
        
        # *  (batch, 4, height / 8, width / 8) -> same
        stdev = torch.exp(0.5 * log_var)
        
        # * z = mean + stdev * eps, eps <- N(0 ,I)
        # * (batch, 4, height / 8, width / 8)
        x = mean + stdev * noise
        
        
        # * Sacle the output by a constant
        # * 起到一定归一化的作用
        x *= 0.18215
        
        # * (batch, 4, height / 8, width / 8)
        return x