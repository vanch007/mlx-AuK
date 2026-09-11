import math
import mlx.core as mx
import mlx.nn as nn
from ..config import Flux2EditConfig
from .modules import (
    AdaLayerNorm,
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    SwiGLUFeedForward,
    TimestepEmbedding,
    apply_rotary_emb,
)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight

class MMDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float = 2.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        self.attn_norm_c = AdaLayerNorm(dim)
        self.attn_norm_x = AdaLayerNorm(dim)

        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=False)
        self.to_qkv_c = nn.Linear(dim, heads * dim_head * 3, bias=False)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)
        self.c_q_norm = RMSNorm(dim_head)
        self.c_k_norm = RMSNorm(dim_head)

        self.to_out = nn.Linear(heads * dim_head, dim)
        self.to_out_c = nn.Linear(heads * dim_head, dim)

        self.ff_norm_c = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff_c = SwiGLUFeedForward(dim=dim, mult=ff_mult)

        self.ff_norm_x = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff_x = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def __call__(self, x: mx.array, c: mx.array, t: mx.array, rope: mx.array = None, c_rope: mx.array = None):
        B, N, _ = x.shape
        _, Nc, _ = c.shape

        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        qkv = self.to_qkv(norm_x).reshape(B, N, 3, self.heads, self.dim_head)
        q, k, v = mx.split(qkv, 3, axis=2)
        q = q.squeeze(2).swapaxes(1, 2)
        k = k.squeeze(2).swapaxes(1, 2)
        v = v.squeeze(2).swapaxes(1, 2)

        c_qkv = self.to_qkv_c(norm_c).reshape(B, Nc, 3, self.heads, self.dim_head)
        c_q, c_k, c_v = mx.split(c_qkv, 3, axis=2)
        c_q = c_q.squeeze(2).swapaxes(1, 2)
        c_k = c_k.squeeze(2).swapaxes(1, 2)
        c_v = c_v.squeeze(2).swapaxes(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        c_q = self.c_q_norm(c_q)
        c_k = self.c_k_norm(c_k)

        if rope is not None:
            q = apply_rotary_emb(q, rope)
            k = apply_rotary_emb(k, rope)
        if c_rope is not None:
            c_q = apply_rotary_emb(c_q, c_rope)
            c_k = apply_rotary_emb(c_k, c_rope)

        full_k = mx.concatenate([c_k, k], axis=2)
        full_v = mx.concatenate([c_v, v], axis=2)

        scale = 1.0 / math.sqrt(self.dim_head)
        c_attn = mx.fast.scaled_dot_product_attention(c_q, full_k, full_v, scale=scale)
        x_attn = mx.fast.scaled_dot_product_attention(q, full_k, full_v, scale=scale)

        c_attn = c_attn.swapaxes(1, 2).reshape(B, Nc, self.heads * self.dim_head)
        x_attn = x_attn.swapaxes(1, 2).reshape(B, N, self.heads * self.dim_head)

        c_out = self.to_out_c(c_attn)
        x_out = self.to_out(x_attn)

        c = c + c_gate_msa[:, None, :] * c_out
        x = x + x_gate_msa[:, None, :] * x_out

        norm_c = self.ff_norm_c(c) * (1.0 + c_scale_mlp[:, None, :]) + c_shift_mlp[:, None, :]
        c = c + c_gate_mlp[:, None, :] * self.ff_c(norm_c)

        norm_x = self.ff_norm_x(x) * (1.0 + x_scale_mlp[:, None, :]) + x_shift_mlp[:, None, :]
        x = x + x_gate_mlp[:, None, :] * self.ff_x(norm_x)

        return c, x

class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float = 2.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        self.attn_norm = AdaLayerNorm(dim)
        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=False)
        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)
        self.to_out = nn.Linear(heads * dim_head, dim)

        self.ff_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def __call__(self, x: mx.array, t: mx.array, rope: mx.array = None):
        B, N, _ = x.shape
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        qkv = self.to_qkv(norm).reshape(B, N, 3, self.heads, self.dim_head)
        q, k, v = mx.split(qkv, 3, axis=2)
        q = q.squeeze(2).swapaxes(1, 2)
        k = k.squeeze(2).swapaxes(1, 2)
        v = v.squeeze(2).swapaxes(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = apply_rotary_emb(q, rope)
            k = apply_rotary_emb(k, rope)

        scale = 1.0 / math.sqrt(self.dim_head)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        attn = attn.swapaxes(1, 2).reshape(B, N, self.heads * self.dim_head)
        out = self.to_out(attn)

        x = x + gate_msa[:, None, :] * out

        norm_x = self.ff_norm(x) * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.ff(norm_x)

        return x

class Flux2Edit(nn.Module):
    def __init__(self, cfg: Flux2EditConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.latent_dim = cfg.latent_dim

        self.audio_linear = nn.Linear(cfg.latent_dim, cfg.dim)
        self.audio_pos = ConvPositionEmbedding(cfg.dim)
        self.text_linear = nn.Linear(cfg.text_hidden_dim, cfg.dim)

        self.time_embed = TimestepEmbedding(cfg.dim)

        self.double_blocks = [
            MMDiTBlock(cfg.dim, cfg.heads, cfg.dim_head, ff_mult=cfg.ff_mult)
            for _ in range(cfg.num_layers)
        ]
        self.single_blocks = [
            DiTBlock(cfg.dim, cfg.heads, cfg.dim_head, ff_mult=cfg.ff_mult)
            for _ in range(cfg.num_single_layers)
        ]

        self.norm_out = AdaLayerNorm_Final(cfg.dim)
        self.proj_out = nn.Linear(cfg.dim, cfg.latent_dim)

    def _embed_audio(self, x: mx.array) -> mx.array:
        # x is [B, N, latent_dim]
        h = self.audio_linear(x)
        return self.audio_pos(h) + h

    def __call__(
        self,
        x: mx.array,
        text: mx.array,
        time: mx.array,
        ref: mx.array = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
    ) -> mx.array:
        # x: current noisy target latents [B, N_target, D_latent]
        # ref: reference latents [B, N_ref, D_latent]
        if ref is not None and ref.shape[1] > 0:
            if drop_audio_cond:
                ref = mx.zeros_like(ref)
            full_audio = mx.concatenate([ref, x], axis=1)
        else:
            full_audio = x

        x_emb = self._embed_audio(full_audio)
        c_emb = self.text_linear(text)
        if drop_text:
            c_emb = mx.zeros_like(c_emb)

        t_emb = self.time_embed(time)

        # compute RoPE frequencies for sequence
        head_dim = self.cfg.dim_head
        theta = 10000.0
        dim_pairs = head_dim // 2
        freqs_seq = mx.arange(x_emb.shape[1] + c_emb.shape[1] + 4096)
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim_pairs) / dim_pairs))
        rope_freqs = freqs_seq[:, None] * inv_freq[None, :]

        rope_x = rope_freqs[:x_emb.shape[1]]
        rope_c = rope_freqs[:c_emb.shape[1]]

        c, h = c_emb, x_emb
        for block in self.double_blocks:
            c, h = block(h, c, t_emb, rope=rope_x, c_rope=rope_c)

        # Single stream: concatenate [c, h]
        N_c = c.shape[1]
        joint = mx.concatenate([c, h], axis=1)
        rope_joint = rope_freqs[:joint.shape[1]]

        for block in self.single_blocks:
            joint = block(joint, t_emb, rope=rope_joint)

        # extract audio tokens back out
        h_out = joint[:, N_c:, :]
        # extract target portion (excluding ref prefix)
        N_ref = ref.shape[1] if (ref is not None and ref.shape[1] > 0) else 0
        target_out = h_out[:, N_ref:, :]

        target_norm = self.norm_out(target_out, emb=t_emb)
        pred = self.proj_out(target_norm)
        return pred
