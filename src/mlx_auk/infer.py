import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

# Ensure upstream auk modules are importable
UPSTREAM_SRC = "/tmp/AuK_upstream/src"
if UPSTREAM_SRC not in sys.path:
    sys.path.insert(0, UPSTREAM_SRC)

from auk.infer.infer_auk import AukInfer as UpstreamAukInfer


class AukInfer:
    def __init__(
        self,
        config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        qwen_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ):
        base_dir = "/Users/vanch/mlx-AuK"
        self.config_path = config_path or os.path.join(base_dir, "ckpts/AuK-Flash/config.yaml")
        self.ckpt_path = ckpt_path or os.path.join(base_dir, "ckpts/AuK-Flash/auk_flash.safetensors")
        self.qwen_path = qwen_path or os.path.join(base_dir, "ckpts/Qwen2.5-Omni-3B")
        self.device = device
        self.target_sample_rate = 24000

        print("Loading full-parity AuK inference engine on %s..." % self.device)
        self.engine = UpstreamAukInfer(
            config_path=self.config_path,
            ckpt_path=self.ckpt_path,
            qwen_path=self.qwen_path,
            device=self.device,
        )
        print("Engine loaded successfully!")

    def generate(
        self,
        messages: List[Dict[str, any]],
        *,
        audio: Optional[Union[str, Tuple[torch.Tensor, int]]] = None,
        gen_seconds: Optional[float] = None,
        nfe: int = 4,
        cfg_strength: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, Dict[str, float]]:
        t_start = time.perf_counter()

        audio_arg = None
        audio_path = audio if isinstance(audio, str) else None

        if audio_path is None and audio is None:
            for m in messages:
                if m.get("role") == "user":
                    for c in m.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "audio":
                            audio_path = c.get("audio") or c.get("audio_url")
                            break

        if audio_path and os.path.exists(audio_path):
            wav_np, sr = sf.read(audio_path)
            if wav_np.ndim == 2:
                wav_np = wav_np.mean(axis=-1)
            t_audio = torch.from_numpy(wav_np).float().unsqueeze(0)
            audio_arg = (t_audio, sr)
        elif isinstance(audio, tuple):
            audio_arg = audio

        wav, sr = self.engine.generate(
            messages,
            audio=audio_arg,
            gen_seconds=gen_seconds,
            seed=seed,
        )

        total_latency = time.perf_counter() - t_start
        wav_np = wav.squeeze().cpu().numpy()
        duration = len(wav_np) / float(sr)
        rtf = total_latency / duration if duration > 0 else 0.0

        metrics = {
            "latency": total_latency,
            "audio_duration": duration,
            "rtf": rtf,
            "sample_rate": sr,
        }
        return wav_np, sr, metrics


def save_audio(waveform: np.ndarray, sample_rate: int, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, waveform, sample_rate)
