from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BigVGANConfig:
    upsample_rates: List[int] = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [10, 8, 6, 4, 4, 4])
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    downsample_rates: List[int] = field(default_factory=lambda: [2, 2, 2, 3, 4, 5])
    downsample_channels: List[int] = field(default_factory=lambda: [12, 24, 48, 96, 192, 384, 768])
    snake_logscale: bool = True
    latent_dim: int = 64
    target_sample_rate: int = 24000
    downsample_rate: int = 480
    use_vae: bool = True
    causal: bool = True
    flow_hidden_channels: int = 256
    act_causal: bool = True

@dataclass
class Flux2EditConfig:
    dim: int = 1536
    depth: int = 8
    heads: int = 24
    dim_head: int = 64
    dropout: float = 0.0
    ff_mult: float = 2.0
    text_hidden_dim: int = 2048
    num_layers: int = 10
    num_single_layers: int = 20
    latent_dim: int = 64

@dataclass
class AuKConfig:
    name: str = "AuK-Flash"
    is_flash: bool = True
    dit: Flux2EditConfig = field(default_factory=Flux2EditConfig)
    vae: BigVGANConfig = field(default_factory=BigVGANConfig)
    text_encoder_path: str = "ckpts/Qwen2.5-Omni-3B"
    t_sampling: str = "logistic_normal"
    P_mean: float = -0.8
    P_std: float = 0.8
    flash_t_grid: List[float] = field(default_factory=lambda: [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0])
    default_nfe: int = 4
    default_cfg: float = 0.0

    @classmethod
    def auk_flash(cls):
        return cls(name="AuK-Flash", is_flash=True, default_nfe=4, default_cfg=0.0)

    @classmethod
    def auk_base(cls):
        return cls(name="AuK", is_flash=False, default_nfe=32, default_cfg=2.0)
