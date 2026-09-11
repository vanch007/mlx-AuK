import math
import mlx.core as mx
import mlx.nn as nn

class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        freqs = mx.exp(mx.arange(half_dim) * -emb)
        args = scale * x[:, None] * freqs[None, :]
        return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.linear1 = nn.Linear(freq_embed_dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def __call__(self, t: mx.array) -> mx.array:
        # t is 1D or scalar [B]
        if t.ndim == 0:
            t = t[None]
        emb = self.time_embed(t)
        emb = nn.silu(self.linear1(emb))
        return self.linear2(emb)

class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.linear = nn.Linear(dim, dim * 6)

    def __call__(self, x: mx.array, emb: mx.array):
        # emb: [B, dim]
        mod = self.linear(nn.silu(emb)) # [B, dim * 6]
        # split into 6 chunks
        chunks = mx.split(mod, 6, axis=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        # elementwise_affine=False LayerNorm
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        norm_x = (x - mean) / mx.sqrt(var + self.eps)

        norm_x = norm_x * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.linear = nn.Linear(dim, dim * 2)

    def __call__(self, x: mx.array, emb: mx.array):
        mod = self.linear(nn.silu(emb))
        shift, scale = mx.split(mod, 2, axis=-1)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        norm_x = (x - mean) / mx.sqrt(var + self.eps)
        return norm_x * (1.0 + scale[:, None, :]) + shift[:, None, :]

class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * mult * 2 / 3)
        self.w12 = nn.Linear(dim, hidden_dim * 2)
        self.w3 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        proj = self.w12(x)
        x1, x2 = mx.split(proj, 2, axis=-1)
        act = x1 * nn.silu(x2)
        return self.w3(act)

class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        # x is [B, N, D]
        res = x
        x = nn.mish(self.conv1(x))
        x = nn.mish(self.conv2(x))
        return x + res

def apply_rotary_emb(x: mx.array, freqs: mx.array) -> mx.array:
    # x is [B, Heads, N, HeadDim]
    # freqs is [N, HeadDim // 2]
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    cos = mx.cos(freqs)[None, None, :x.shape[2], :]
    sin = mx.sin(freqs)[None, None, :x.shape[2], :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return mx.concatenate([rx1, rx2], axis=-1)
