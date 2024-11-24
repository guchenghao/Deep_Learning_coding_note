import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = 512 // 8
LATENT_HEIGHT = 512 // 8


# * strength表示生成的图片与原始图片的相关程度
# * do_CFG表示是否采用classifier-free Gudience
# * cfg_scale表示生成过程关注prompt的强度(1-14)
# * n_interface_steps表示时间步数
# * prompt描述的是希望生成什么样的图片；uncond_prmopt是一个空字符串
def generate(
    prompt,
    uncond_prompt,
    input_image,
    strength=0.8,
    do_CFG=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):

    # ! 将整个前向过程都包含在无梯度计算中
    # * torch.no_grad() 会减少显存消耗
    # * 在 torch.no_grad() 上下文中，PyTorch 会临时关闭张量的自动求导机制（autograd），避免记录用于计算梯度的操作
    with torch.no_grad():

        if strength < 0 or strength > 1:
            raise ValueError("Strength must be betwwen 0 and 1")

        # * 这个idle_device便于在处理数据时，将不使用的数据放置到闲置的设备(GPU或者CPU)中
        if idle_device:
            to_idle: lambda x: x.to(idle_device)

        else:
            to_idle: lambda x: x

        # * 控制随机数生成的状态，从而确保随机操作的可重复性
        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()

        else:

            generator.manual_seed(seed=seed)

        clip = models["clip"]
        clip.to(device)
        
        # * 根据是够采用CFG准则，来判断如何构建token
        if do_CFG:

            # * Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encoder_plus([prompt], padding="max_length", max_length=77).input_ids

            # * (batch_size, seq_length)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # * (batch_size, seq_length) -> (batch_size, seq_length, hiddendim=768)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encoder_plus([uncond_prompt], padding="max_length", max_length=77).input_ids

            # * (batch_size, seq_length)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # * (batch_size, seq_length) -> (batch_size, seq_length, hiddendim=768)
            uncond_context = clip(uncond_tokens)

            # * total context (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])

        else:
            # * Convert it into a list of tokens
            tokens = tokenizer.batch_encoder_plus([prompt], padding="max_length", max_length=77).input_ids

            # * (batch_size, seq_length)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # * (1, 77, 768)
            # * 只使用prompt，不考虑uncondition的情况
            context = clip(tokens)

        to_idle(clip)  # * 因为clip已经使用完了，可以将clip模型的参数放置到闲置的设备中，避免占用计算资源

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_step(n_inference_steps)  # * 设置生成过程的时间步数

        else:
            raise ValueError(f"Unknow sampler {sampler_name}")

        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)  # * 潜在变量的shape

        if input_image:
            # * 处理image-to-image的情况
            encoder = models["encoder"]  # * VAE Encoder
            encoder.to(device)

            input_image_tensor = input_image.resize(WIDTH, HEIGHT)
            input_image_tensor = np.array(input_image_tensor)
            # * (height, width, channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # * (height, width, channel) -> (batch, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # * (batch, height, width, channel) -> (batch, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # * 重参数化技巧
            encoder_noise_eps = torch.randn(size=latents_shape, device=device, generator=generator)

            # * encoder image to latent
            latent_image = encoder(input_image_tensor, encoder_noise_eps)

            # * strength的大小决定了添加noise的多少, 例如strength设置为1，说明初始噪声强度是最大的
            sampler.set_strength(strength=strength)  
            latent_image = sampler.add_noise(latent_image, sampler.timesteps[0])

            to_idle(encoder)

        else:
            # * 处理text-to-image的情况
            # * 此时没有image输入，直接从random noise开始去噪, latents from normal distribution N(0, I)
            latents = torch.randn(
                size=latents_shape, device=device, generator=generator
            )
            
        
        # * 进入去噪阶段
        # * 1000 980 960 940 920 900 880 860 840 .... (总共50步), 1000 / 20
        diffusion = models["diffusion"]
        diffusion.to(device)
        
        
        
        
        
        
        
        
            
            
