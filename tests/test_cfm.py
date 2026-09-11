import sys
sys.path.insert(0, "/Users/vanch/mlx-AuK/src")

import mlx.core as mx
from mlx_auk.config import Flux2EditConfig
from mlx_auk.dit.flux2 import Flux2Edit
from mlx_auk.dit.cfm import CFMEdit

def test_cfm_sampling():
    cfg = Flux2EditConfig(dim=256, heads=4, dim_head=64, num_layers=2, num_single_layers=2, latent_dim=64, text_hidden_dim=512)
    model = Flux2Edit(cfg)
    cfm = CFMEdit(model, num_channels=64)

    ref = mx.random.normal((1, 20, 64))
    text = mx.random.normal((1, 30, 512))

    t_grid = [0.0, 0.07612, 0.29289, 0.61731, 1.0] # AuK-Flash 4-step
    full, target = cfm.sample(
        cond=ref,
        text_embed=text,
        target_len=50,
        t_grid=t_grid,
        cfg_strength=0.0,
    )
    assert target.shape == (1, 50, 64)
    assert full.shape == (1, 70, 64)
    print("CFM 4-step sampling test passed!")

if __name__ == "__main__":
    test_cfm_sampling()
