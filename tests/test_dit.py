import sys
sys.path.insert(0, "/Users/vanch/mlx-AuK/src")

import mlx.core as mx
from mlx_auk.config import Flux2EditConfig
from mlx_auk.dit.modules import TimestepEmbedding, AdaLayerNorm, AdaLayerNorm_Final
from mlx_auk.dit.flux2 import MMDiTBlock, DiTBlock, Flux2Edit

def test_dit_blocks():
    dim = 256
    heads = 4
    dim_head = 64
    t_embed = TimestepEmbedding(dim)
    t = t_embed(mx.array([0.5]))

    # MMDiT
    mmdit = MMDiTBlock(dim, heads, dim_head)
    x = mx.random.normal((1, 20, dim))
    c = mx.random.normal((1, 30, dim))
    c_out, x_out = mmdit(x, c, t)
    assert c_out.shape == (1, 30, dim)
    assert x_out.shape == (1, 20, dim)

    # Single DiT
    dit = DiTBlock(dim, heads, dim_head)
    joint = mx.concatenate([c_out, x_out], axis=1)
    j_out = dit(joint, t)
    assert j_out.shape == (1, 50, dim)
    print("DiT block tests passed!")

def test_flux2_forward():
    cfg = Flux2EditConfig(dim=256, heads=4, dim_head=64, num_layers=2, num_single_layers=2, latent_dim=64, text_hidden_dim=512)
    model = Flux2Edit(cfg)

    x = mx.random.normal((1, 50, 64))
    text = mx.random.normal((1, 30, 512))
    t = mx.array([0.25])
    ref = mx.random.normal((1, 20, 64))

    v = model(x, text, t, ref=ref)
    assert v.shape == (1, 50, 64)
    print("Flux2Edit forward test passed!")

if __name__ == "__main__":
    test_dit_blocks()
    test_flux2_forward()
