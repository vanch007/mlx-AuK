from typing import Optional, List, Tuple
import math
import mlx.core as mx
import mlx.nn as nn
from ..config import BigVGANConfig
from .modules import Conv1d, ConvTranspose1d, AMPBlock1
from .activations import SnakeBeta, Activation1d

class ResStack(nn.Module):
    def __init__(self, channel: int, kernel_size: int = 3, base: int = 3, nums: int = 4):
        super().__init__()
        self.conv1 = [Conv1d(channel, channel, kernel_size=kernel_size, dilation=base**i) for i in range(nums)]
        self.conv2 = [Conv1d(channel, channel, kernel_size=kernel_size, dilation=1) for i in range(nums)]

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2 in zip(self.conv1, self.conv2):
            res = nn.leaky_relu(x, negative_slope=0.2)
            res = c1(res)
            res = nn.leaky_relu(res, negative_slope=0.2)
            res = c2(res)
            x = x + res
        return x

class AudioEncoder(nn.Module):
    def __init__(self, cfg: BigVGANConfig):
        super().__init__()
        self.cfg = cfg
        out_channels = cfg.latent_dim * 2 if cfg.use_vae else cfg.latent_dim
        
        self.pre_conv = Conv1d(1, cfg.downsample_channels[0], kernel_size=3, stride=1)
        self.stages = []
        for (in_c, out_c), down_f in zip(zip(cfg.downsample_channels[:-1], cfg.downsample_channels[1:]), cfg.downsample_rates):
            self.stages.append((
                Conv1d(in_c, out_c, kernel_size=down_f * 2, stride=down_f),
                ResStack(out_c, kernel_size=3, base=2, nums=6),
            ))
        self.post_conv = Conv1d(cfg.downsample_channels[-1], out_channels, kernel_size=3, stride=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, 1, T]
        x = nn.leaky_relu(self.pre_conv(x), negative_slope=0.2)
        for conv, res in self.stages:
            x = conv(x)
            x = res(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
        x = self.post_conv(x)
        return x

class BigVGANFlowVAE(nn.Module):
    def __init__(self, cfg: BigVGANConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.hop_size = math.prod(cfg.downsample_rates)
        self.global_mean = mx.zeros((cfg.latent_dim,))
        self.global_log_std = mx.ones((cfg.latent_dim,))

        self.audio_encoder = AudioEncoder(cfg)

        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.upsample_rates)

        self.conv_pre = Conv1d(cfg.latent_dim, cfg.upsample_initial_channel, kernel_size=7, stride=1, causal=False)

        self.ups = []
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            in_ch = cfg.upsample_initial_channel // (2**i)
            out_ch = cfg.upsample_initial_channel // (2**(i + 1))
            self.ups.append([ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=u, causal=cfg.causal)])

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = cfg.upsample_initial_channel // (2**(i + 1))
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(ch, kernel_size=k, dilation=d, causal=cfg.causal, snake_logscale=cfg.snake_logscale))

        out_ch = cfg.upsample_initial_channel // (2**len(cfg.upsample_rates))
        self.activation_post = Activation1d(SnakeBeta(out_ch, alpha_logscale=cfg.snake_logscale))
        self.conv_post = Conv1d(out_ch, 1, kernel_size=7, stride=1, causal=False)

    def encoding_and_normalization(self, sample: mx.array, sample_lengths: Optional[int] = None):
        # sample: [B, 1, T]
        latent_stats = self.audio_encoder(sample)
        # chunk 2 on channel dim: [B, D*2, T_latent]
        mean = latent_stats[:, :self.latent_dim, :]
        latents = mx.transpose(mean, (0, 2, 1)) # [B, T_latent, D]
        latents = (latents - self.global_mean[None, None, :]) / mx.sqrt(self.global_log_std[None, None, :])
        return latents

    def denormalize(self, latents: mx.array) -> mx.array:
        # latents: [B, T, D]
        return latents * mx.sqrt(self.global_log_std[None, None, :]) + self.global_mean[None, None, :]

    def inference_from_latents(self, x: mx.array) -> mx.array:
        # x is [B, D, T_latent]
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                out = rb(x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = mx.clip(x, -1.0, 1.0)
        return x
