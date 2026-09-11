import os
import json
import torch
from safetensors.torch import load_file, save_file

SOURCE_AUK = "/Users/vanch/mlx-AuK/ckpts/AuK-Flash/auk_flash.safetensors"
SOURCE_VAE = "/Users/vanch/mlx-AuK/ckpts/AuK-Flash/vae.safetensors"
OUTPUT_DIR = "/Users/vanch/mlx-AuK/models/mlx-auk-flash"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Converting DiT backbone to MLX native safetensors...")
st = load_file(SOURCE_AUK)
mlx_dit = {}
for k, v in st.items():
    clean_k = k.replace("ema_model.", "")
    if not clean_k.startswith("text_encoder."):
        mlx_dit[clean_k] = v.contiguous().to(torch.float32)

dit_out_path = os.path.join(OUTPUT_DIR, "dit.safetensors")
save_file(mlx_dit, dit_out_path)
print("Saved %d tensors to %s (%.2f GB)" % (len(mlx_dit), dit_out_path, os.path.getsize(dit_out_path)/(1024**3)))

print("2. Converting BigVGAN VAE to MLX native safetensors...")
st_vae = load_file(SOURCE_VAE)
mlx_vae = {}
for k, v in st_vae.items():
    mlx_vae[k] = v.contiguous().to(torch.float32)

vae_out_path = os.path.join(OUTPUT_DIR, "vae.safetensors")
save_file(mlx_vae, vae_out_path)
print("Saved %d tensors to %s (%.1f MB)" % (len(mlx_vae), vae_out_path, os.path.getsize(vae_out_path)/(1024**2)))

cfg_dict = {
    "model_type": "auk-flash",
    "is_flash": True,
    "dim": 1536,
    "heads": 24,
    "num_layers": 10,
    "num_single_layers": 20,
    "latent_dim": 64,
    "sample_rate": 24000,
    "downsample_rate": 480,
    "t_grid": [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0],
}
cfg_path = os.path.join(OUTPUT_DIR, "config.json")
with open(cfg_path, "w") as f:
    json.dump(cfg_dict, f, indent=2)
print("Saved config to %s" % cfg_path)
print("MLX weight conversion completed successfully!")
