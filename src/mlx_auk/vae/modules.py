import math
import mlx.core as mx
import mlx.nn as nn
from .activations import SnakeBeta, Activation1d

class Conv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, padding: int = 0, causal: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.causal = causal
        
        # In PyTorch weight is [out_channels, in_channels, kernel_size]
        # In MLX conv1d weight is [out_channels, kernel_size, in_channels]
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(-scale, scale, (out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x input is [B, C, T]
        # Transpose to [B, T, C] for MLX conv1d
        x_t = mx.transpose(x, (0, 2, 1))
        
        effective_k = (self.kernel_size - 1) * self.dilation + 1
        if self.causal:
            # left pad
            pad_left = effective_k - self.stride
            if pad_left > 0:
                x_t = mx.pad(x_t, [(0, 0), (pad_left, 0), (0, 0)])
            pad = 0
        else:
            pad = self.padding if self.padding > 0 else (effective_k - 1) // 2
        
        out = mx.conv1d(x_t, self.weight, stride=self.stride, padding=pad, dilation=self.dilation)
        out = out + self.bias[None, None, :]
        # Transpose back to [B, C, T]
        return mx.transpose(out, (0, 2, 1))

class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, causal: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        
        # PyTorch ConvTranspose1d weight: [in_channels, out_channels, kernel_size]
        # MLX conv_transpose1d weight: [out_channels, kernel_size, in_channels]
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(-scale, scale, (out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, C, T] -> [B, T, C]
        x_t = mx.transpose(x, (0, 2, 1))
        pad = (self.kernel_size - self.stride) // 2
        out = mx.conv_transpose1d(x_t, self.weight, stride=self.stride, padding=pad)
        out = out + self.bias[None, None, :]
        return mx.transpose(out, (0, 2, 1))

class AMPBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5), causal: bool = True, snake_logscale: bool = True):
        super().__init__()
        self.convs1 = [
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], causal=causal),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], causal=causal),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], causal=causal),
        ]
        self.convs2 = [
            Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal),
            Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal),
            Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal),
        ]
        self.activations = [
            Activation1d(SnakeBeta(channels, alpha_logscale=snake_logscale))
            for _ in range(len(self.convs1) + len(self.convs2))
        ]

    def __call__(self, x: mx.array) -> mx.array:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x
