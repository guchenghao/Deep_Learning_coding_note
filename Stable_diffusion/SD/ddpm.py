import torch
import numpy as np


class DDPMSampler:
    """Some Information about DDPMSampler"""

    def __init__(
        self, generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120
    ):
        # * 主要是用于生成论文中的beta参数序列，因为前向扩散过程的beta是一个序列
        # * 在stable diffusion中主要是采用线性采样的方式来生成序列，也可以使用余弦或者正弦函数来生成
        # * linear scale schedule
        # * (1000,)
        self.betas = torch.linspace(
            beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
        )

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

        # * [980, 960, 940, 920, ..., 0], len = 1000 // 20 = 50
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def set_strength(self, strength=1):
        # * strength参数表示UNET网络在去噪过程中对原始图像修改的自由度
        # * strength越大，表示加的噪声就越多，UNET的修改自由度越高； strength越小，表示加的噪声越少，UNET的修改自由度越小
        # * 如果strength=0.8, strat_step = 10,表示从timesteps数组中第10个时间步开始去噪；如果strength=1, strat_step = 0，表示从头开始去噪
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def _get_previous_timestep(self, timestep):
        # * 如果当前timestep是980， timestep_prev = 980 - 20 = 960
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)

        return prev_t

    def _get_variance(self, timestep):
        # * 计算q(Xt-1|Xt, X0)的variance
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]

        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev

        current_beta_t = 1 - current_alpha_t

        current_variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        current_variance = torch.clamp(current_variance, min=1e-20)  # * 确保variance不等于0

        return current_variance

    def step(self, timestep, latents, model_output):
        # * Understand Diffusion Model文档说明了loss function的3种视角的解读：KL散度：D_KL(q(Xt-1| Xt, X0) || p(Xt-1|Xt)); 数据重构误差: mse(Xθ - X0); 噪声重构误差：mse(Epsθ - Eps)
        # * step函数中的计算是基于第2中视角(数据重构误差)
        # * model_output就是UNET预测出的noise
        # * 接下来都在计算lossfunction中的P(Xt-1|Xt)的mean和variance，这个是可以直接计算
        t = timestep

        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[timestep]

        # * 处理边界条件： 当前时间步=0时，将alpha_prod_t_prev设置为1， 这样beta_prod_t_prev就为0了
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        # * 这些计算为了计算论文中sampling伪代码中，均值前的系数
        # * 之后的论文中也指出了这些系数可以去掉，因为去掉效果更好
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # * Compute the predicted original sample(原始图片) using formular (15) in DDPM paper
        # * 反向去噪过程除了可以理解为noise predictor之外，也可以理解为借助神经网络利用潜在张量Xt和timestep t来预测X0
        # * 这是利用预测出来的noise和Xt，再根据前向的公式，反推X0，这里的X0并不真是true data
        predicted_original_sample_x0 = (
            latents - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        predicted_original_sample_x0_coff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t

        current_sample_coff = (current_alpha_t**0.5 * beta_prod_t_prev) / beta_prod_t

        # * mu_theta_bar(Xt, t)
        # * 详细公式可以参考我的SD说明文档
        pred_prev_sample = predicted_original_sample_x0_coff * predicted_original_sample_x0 + current_sample_coff * latents

        # * 在生成的最后一步，不需要添加variance
        current_std = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, device=device, generator=self.generator, dtype=model_output.dtype)
            current_std = self._get_variance(t) ** 0.5 * noise

        # * 重参数化技巧
        # * 根据我们前面计算的mean和variance，从P(Xt-1|Xt)中sample出 Xt-1
        pred_prev_sample = pred_prev_sample + current_std 

        return pred_prev_sample

    def add_noise(self, original_samples, timesteps):
        # * 添加噪声，实际上不需要一步一步地去添加噪声，我们只需要知道时间步，直接添加相应时间步对应量的噪声即可
        # * 需要使用重参数化技巧

        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )

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
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        noise_samples = mean + standard_deviation * noise

        return noise_samples
