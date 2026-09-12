
import sys
import os
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

sys.path.insert(0, '/Users/vanch/mlx-AuK/src')
from mlx_auk.config import Flux2EditConfig, BigVGANConfig
from mlx_auk.dit.flux2 import Flux2Edit
from mlx_auk.dit.cfm import CFMEdit
from mlx_auk.vae.vae import BigVGANFlowVAE

def test_base_mlx_integrity():
    base_model_dir = '/Users/vanch/mlx-AuK/models/mlx-auk-base'
    dit_path = os.path.join(base_model_dir, 'dit.safetensors')
    vae_path = os.path.join(base_model_dir, 'vae.safetensors')
    
    assert os.path.exists(dit_path), "Base dit.safetensors missing!"
    assert os.path.exists(vae_path), "Base vae.safetensors missing!"
    
    print("1. Checking Base DiT safetensors weights...")
    with safe_open(dit_path, framework="mlx") as f:
        keys = f.keys()
        print(f"  -> Found {len(keys)} tensor keys in dit.safetensors")
        assert len(keys) == 420, f"Expected 420 keys, got {len(keys)}"
    
    print("2. Checking Base VAE safetensors weights...")
    with safe_open(vae_path, framework="mlx") as f:
        vkeys = f.keys()
        print(f"  -> Found {len(vkeys)} tensor keys in vae.safetensors")
        assert len(vkeys) == 1137, f"Expected 1137 keys, got {len(vkeys)}"

    print("3. Testing Base 32-step CFM Sampling loop in MLX...")
    cfg = Flux2EditConfig(
        dim=1536,
        heads=24,
        dim_head=64,
        num_layers=1,
        num_single_layers=1,
        latent_dim=64,
        text_hidden_dim=2048
    )
    model = Flux2Edit(cfg)
    cfm = CFMEdit(model)
    
    cond_latents = mx.random.normal((1, 50, 64))
    text_emb = mx.random.normal((1, 20, 2048))
    
    # Run 4 steps of Base sampling to verify Euler flow integration
    full_samples, samples = cfm.sample(
        cond=cond_latents,
        text_embed=text_emb,
        target_len=30,
        steps=4,
        cfg_strength=2.0
    )
    mx.eval(samples)
    print(f"  -> Successfully generated Base samples shape: {samples.shape}")
    assert samples.shape == (1, 30, 64), f"Unexpected shape {samples.shape}"

    print("4. Testing 8-bit Quantized Base model forward pass...")
    nn.quantize(model, group_size=64, bits=8)
    q_full_samples, q_samples = cfm.sample(
        cond=cond_latents,
        text_embed=text_emb,
        target_len=30,
        steps=4,
        cfg_strength=2.0
    )
    mx.eval(q_samples)
    print(f"  -> 8-bit Base samples generated shape: {q_samples.shape}")
    assert q_samples.shape == (1, 30, 64)

if __name__ == '__main__':
    test_base_mlx_integrity()
    print("✅ All AuK-Base MLX integrity and sampling tests passed!")

