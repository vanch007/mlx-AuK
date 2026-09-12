
import sys
sys.path.insert(0, '/Users/vanch/mlx-AuK/src')

import mlx.core as mx
import mlx.nn as nn
from mlx_auk.config import Flux2EditConfig
from mlx_auk.dit.flux2 import Flux2Edit

def test_quantization():
    cfg = Flux2EditConfig(
        dim=1536,
        heads=24,
        dim_head=64,
        num_layers=1,
        num_single_layers=1,
        latent_dim=64,
        text_hidden_dim=2048
    )
    
    # 8-bit test
    model_8 = Flux2Edit(cfg)
    nn.quantize(model_8, group_size=64, bits=8)
    x = mx.random.normal((1, 50, 64))
    text = mx.random.normal((1, 20, 2048))
    t = mx.array([0.25])
    v8 = model_8(x, text, t)
    mx.eval(v8)
    assert v8.shape == (1, 50, 64)
    print("✅ 8-bit quantization test passed!")

    # 4-bit test
    model_4 = Flux2Edit(cfg)
    nn.quantize(model_4, group_size=64, bits=4)
    v4 = model_4(x, text, t)
    mx.eval(v4)
    assert v4.shape == (1, 50, 64)
    print("✅ 4-bit quantization test passed!")

if __name__ == '__main__':
    test_quantization()
    print("All quantization tests passed successfully!")

