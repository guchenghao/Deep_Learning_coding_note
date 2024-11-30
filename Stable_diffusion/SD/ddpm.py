import torch
import numpy as np
# import torch.nn as nn


class DDPMSampler():
    """Some Information about DDPMSampler"""
    def __init__(self, generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
        # * 主要是用于生成论文中的beta参数序列，因为前向扩散过程的beta是一个序列
        # * 在stable diffusion中主要是采用线性采样的方式来生成序列，也可以使用余弦或者正弦函数来生成
        # * linear scale schedule
        # * (1000,)
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32)

        self.alphas = 1.0 - self.betas  # * (1000,)

        # * 计算alpha的累计乘积 (1000,)
        # * [1, 1 * 2, 1 * 2 * 3, ...]
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps

        # * [999, 998, ..., 0]
        # * (1000,)
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_steps(self, num_inference_steps=50):
        # * 通常来说，不会真的去inference 1000步，一般只需要inference 50步
        # * 根据输入的inference_steps来调整去噪过程的时间步
        self.num_inference_steps = num_inference_steps

        # * [999, 979, 959, 939, ..., 0]
        # * 步长为20，总共50步
        step_ratio = self.num_training_steps // num_inference_steps

        # * [980, 960, 940, 920, ..., 0]
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, original_samples, timesteps):
        # * 添加噪声，实际上不需要一步一步地去添加噪声，我们只需要知道时间步，直接添加相应时间步对应量的噪声即可
        # * 需要使用重参数化技巧

        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)

        timesteps = timesteps.to(
            device=original_samples.device, dtype=original_samples.dtype
        )

        # * (1,)
        sqrt_aplpha_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_aplpha_cumprod = sqrt_aplpha_cumprod.flatten()

        # * 将sqrt_aplpha_cumprod扩展成宇original_samples一样的维度，这样才能进行相乘
        # * len(...shape)显示这个tensor有几个维度
        while len(sqrt_aplpha_cumprod.shape) < len(original_samples.shape):
            sqrt_aplpha_cumprod = sqrt_aplpha_cumprod.unsqueeze(-1)

        mean = sqrt_aplpha_cumprod * original_samples

        standard_deviation = (1 - alphas_cumprod[timesteps]) ** 0.5
        standard_deviation = standard_deviation.flatten()

        # * 将sqrt_aplpha_cumprod扩展成宇original_samples一样的维度，这样才能进行相加
        while len(standard_deviation.shape) < len(original_samples.shape):
            standard_deviation = standard_deviation.unsqueeze(-1)
        
        
        # * 重参数化技巧
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)

        noise_samples = mean + standard_deviation * noise

        return noise_samples
