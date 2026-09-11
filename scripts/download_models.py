import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

auk_dir = "/Users/vanch/mlx-AuK/ckpts/AuK-Flash"
print("Starting download of tencent/AuK-Flash...")
snapshot_download(
    repo_id="tencent/AuK-Flash",
    local_dir=auk_dir,
    allow_patterns=["*.safetensors", "*.yaml", "*.json"],
)
print("AuK-Flash download complete!")

qwen_dir = "/Users/vanch/mlx-AuK/ckpts/Qwen2.5-Omni-3B"
print("Starting download of Qwen/Qwen2.5-Omni-3B...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-Omni-3B",
    local_dir=qwen_dir,
    allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.model", "*.txt"],
)
print("Qwen2.5-Omni-3B download complete!")
