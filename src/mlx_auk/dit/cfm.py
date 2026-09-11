import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Tuple

class CFMEdit(nn.Module):
    def __init__(
        self,
        transformer,
        num_channels: int = 64,
    ):
        super().__init__()
        self.transformer = transformer
        self.num_channels = num_channels
        self.layer_weights = mx.zeros((28,))
        self.layer_scale = 1.0

    def sample(
        self,
        cond: mx.array, # [B, N_ref, D] reference latents
        text_embed: mx.array, # [B, N_text, D_text]
        target_len: int,
        steps: int = 4,
        cfg_strength: float = 0.0,
        t_grid: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> mx.array:
        B = cond.shape[0] if cond is not None and cond.shape[1] > 0 else 1
        D = self.num_channels

        if seed is not None:
            mx.random.seed(seed)

        # Initial standard normal noise x_0
        x = mx.random.normal((B, target_len, D))

        if t_grid is None:
            # linear grid from 0 to 1
            t_grid = [i / float(steps) for i in range(steps + 1)]

        for i in range(len(t_grid) - 1):
            t_curr = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t_curr

            t_tensor = mx.array([t_curr] * B)

            if cfg_strength < 1e-4:
                v = self.transformer(
                    x=x,
                    text=text_embed,
                    time=t_tensor,
                    ref=cond,
                    drop_audio_cond=False,
                    drop_text=False,
                )
            else:
                v_cond = self.transformer(
                    x=x,
                    text=text_embed,
                    time=t_tensor,
                    ref=cond,
                    drop_audio_cond=False,
                    drop_text=False,
                )
                v_uncond = self.transformer(
                    x=x,
                    text=text_embed,
                    time=t_tensor,
                    ref=cond,
                    drop_audio_cond=True,
                    drop_text=True,
                )
                v = v_uncond + cfg_strength * (v_cond - v_uncond)

            x = x + v * dt

        # Concatenate [ref, generated]
        if cond is not None and cond.shape[1] > 0:
            full = mx.concatenate([cond, x], axis=1)
        else:
            full = x
        return full, x
