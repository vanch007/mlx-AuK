import sys
sys.path.insert(0, "/Users/vanch/mlx-AuK/src")

import mlx.core as mx
from mlx_auk.config import BigVGANConfig
from mlx_auk.vae.activations import SnakeBeta
from mlx_auk.vae.modules import Conv1d, ConvTranspose1d, AMPBlock1
from mlx_auk.vae.vae import BigVGANFlowVAE

def test_snake_beta():
    act = SnakeBeta(64)
    x = mx.random.normal((2, 64, 100))
    out = act(x)
    assert out.shape == (2, 64, 100)

def test_conv1d():
    conv = Conv1d(64, 128, kernel_size=3, padding=1)
    x = mx.random.normal((2, 64, 100))
    out = conv(x)
    assert out.shape == (2, 128, 100)

def test_vae_roundtrip():
    cfg = BigVGANConfig()
    vae = BigVGANFlowVAE(cfg)
    
    # Test audio encoding
    audio = mx.random.normal((1, 1, 24000)) # 1 second of audio
    latents = vae.encoding_and_normalization(audio)
    assert latents.shape[0] == 1
    assert latents.shape[2] == 64
    print("Encoded latent shape:", latents.shape)

    # Test audio decoding
    denorm = vae.denormalize(latents)
    latent_bct = mx.transpose(denorm, (0, 2, 1))
    wav = vae.inference_from_latents(latent_bct)
    assert wav.shape[0] == 1
    assert wav.shape[1] == 1
    print("Decoded audio shape:", wav.shape)

if __name__ == "__main__":
    test_snake_beta()
    test_conv1d()
    test_vae_roundtrip()
    print("All VAE tests passed!")
