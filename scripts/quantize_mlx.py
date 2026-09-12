
import os
import time
import json
import mlx.core as mx
import mlx.nn as nn
import sys

sys.path.insert(0, '/Users/vanch/mlx-AuK/src')
from mlx_auk.config import Flux2EditConfig
from mlx_auk.dit.flux2 import Flux2Edit

BASE_DIR = '/Users/vanch/mlx-AuK'
FP32_PATH = os.path.join(BASE_DIR, 'models/mlx-auk-flash/dit.safetensors')
OUT_8BIT_DIR = os.path.join(BASE_DIR, 'models/mlx-auk-flash-8bit')
os.makedirs(OUT_8BIT_DIR, exist_ok=True)

print('=== 1. Loading Flux2Edit Full Configuration ===')
cfg = Flux2EditConfig(
    dim=1536,
    heads=24,
    dim_head=64,
    num_layers=10,
    num_single_layers=20,
    latent_dim=64,
    text_hidden_dim=2048
)
model = Flux2Edit(cfg)

# Test quantizing model to 8-bit
print('=== 2. Applying 8-bit affine quantization (group_size=64) ===')
t0 = time.time()
nn.quantize(model, group_size=64, bits=8)
print(f'Quantized to 8-bit in {time.time() - t0:.2f}s')

# Save 8-bit weights
weights_8bit = dict(tree_flatten = model.parameters())
out_8bit_path = os.path.join(OUT_8BIT_DIR, 'dit_8bit.safetensors')
flat_weights = {}
def collect(prefix, m):
    for k, v in m.items():
        name = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            collect(name, v)
        elif isinstance(v, mx.array):
            flat_weights[name] = v

collect('', model.parameters())
print(f'Collected {len(flat_weights)} quantized weight arrays.')
mx.save_safetensors(out_8bit_path, flat_weights)
sz_gb = os.path.getsize(out_8bit_path) / (1024**3)
print(f'✅ Saved 8-bit quantized weights to {out_8bit_path} ({sz_gb:.2f} GB)!')

# Measure forward pass memory and latency
print('=== 3. Benchmarking 8-bit DiT Latency on Apple Silicon Metal ===')
x = mx.random.normal((1, 500, 64))
text = mx.random.normal((1, 100, 2048))
t = mx.array([0.25])
ref = mx.random.normal((1, 500, 64))

# Warmup
v = model(x, text, t, ref=ref)
mx.eval(v)

# Latency test over 4 steps (typical Flash sampling)
times = []
for step in range(4):
    t_start = time.perf_counter()
    v = model(x, text, mx.array([step * 0.25]), ref=ref)
    mx.eval(v)
    times.append(time.perf_counter() - t_start)

avg_step_ms = sum(times) / len(times) * 1000
total_4step_s = sum(times)
print(f'✅ 8-bit 4-step sampling compute time: {total_4step_s:.3f}s ({avg_step_ms:.1f}ms / step)!')

