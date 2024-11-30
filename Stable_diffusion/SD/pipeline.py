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
# * prompt描述的是希望生成什么样的图片；uncond_prmopt是一个空字符串 或者negative prompt
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
            context = torch.cat([cond_context, uncond_context], dim=0)

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
            sampler.set_inference_steps(n_inference_steps)  # * 设置生成过程的时间步数

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

            # * 将图片进行Normalization
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
            # * 这里timestep[0]的意思是添加最大时间步对应的噪声，时间步越大说明噪声越多
            latents = sampler.add_noise(latent_image, sampler.timesteps[0])

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

        timesteps = tqdm(sampler.timesteps)

        for i, timestep in enumerate(timesteps):
            # * int -> (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)  # * 这一步采用了transformer中的positional embedding的方法，使用正弦和余弦函数来计算time embedding

            # * (batch, 4, HEIGHT / 8 = 64 , WIDTH / 8 = 64)
            model_input = latents

            if do_CFG:
                # * (batch, 4, HEIGHT / 8 = 64 , WIDTH / 8 = 64) -> (batch * 2, 4, HEIGHT / 8 = 64 , WIDTH / 8 = 64)
                # * 之所以repeat，是因为context中有conditional prompt 和unconditional prompt
                model_input = model_input.repeat(2, 1, 1, 1)

            # * model_output is the prediected noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_CFG:
                output_cond, output_uncond = model_output.chunk(2, dim=0)

                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # * remove the noise prediected by UNET
            # * 移除UNET预测的噪声
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        # * (batch, 4, HEIGHT / 8 = 64 , WIDTH / 8 = 64) -> (batch, 3, HEIGHT, WIDTH)
        images = decoder(latents)
        to_idle(decoder)

        # * (batch, 3, HEIGHT, WIDTH)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # * (batch, 3, HEIGHT, WIDTH) -> (batch, HEIGHT, WIDTH, 3)
        images = images.permute(0, 2, 3, 1)

        images = images.to("cpu", torch.uint8).numpy()  # * 这个操作一般用于图像处理

        return images[0]


# * 用于缩放或归一化原始数据的数值范围
def rescale(x, original_range, target_range, clamp=False):

    original_min, original_max = original_range
    target_min, target_max = target_range
    
    # * x = (x - original_min) / (original_max - original_min)
    # * x = x * (target_max - target_min) + target_min
    # * 这部分代码是先将原始范围缩放到[0, 1]范围，然后在乘上目标范围的宽度，在加上目标范围的下界
    x -= original_min
    x *= (target_max - target_min) / (original_max - original_min)
    x += target_min

    if clamp:
        x = x.clamp(target_min, target_max)
    
    
    return x


# * int -> (1, 320)
# * 将timestep转化为向量
def get_time_embedding(timestep):
    # * 获取time_embedding方式和transformer的positional embedding的方式一模一样
    # * 幂次运算
    # * (160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    
    # * (1, 1) * (1,  160) -> (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    
    
    # * (1, 160) -> (1, 320)
    return torch.cat(torch.cos(x), torch.sin(x), dim=1)
