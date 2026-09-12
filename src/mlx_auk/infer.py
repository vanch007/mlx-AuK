from mlx_auk.anchoring import adapt_edit_instruction
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

UPSTREAM_SRC = "/tmp/AuK_upstream/src"
if UPSTREAM_SRC not in sys.path:
    sys.path.insert(0, UPSTREAM_SRC)

from auk.infer.infer_auk import AukInfer as UpstreamAukInfer

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def estimate_spoken_duration(text: str) -> float:
    if not text:
        return 3.5
    num_zh = len(_CJK_RE.findall(text))
    num_en = len(_EN_WORD_RE.findall(text))
    dur = num_zh * 0.22 + num_en * 0.30
    if dur <= 0.1:
        dur = max(2.5, len(text) * 0.20)
    # Add minor boundary buffer
    return max(1.8, dur + 0.2)


class AukInfer:
    def __init__(
        self,
        config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        qwen_path: Optional[str] = None,
        device: str = "mps",
        repo_id: str = "vanch007/AuK-Flash-MLX",
        **kwargs,
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        mlx_dir = os.path.join(base_dir, "models/mlx-auk-flash")

        # 1. Resolve MLX / PyTorch checkpoint path
        if ckpt_path is None:
            local_dit = os.path.join(mlx_dir, "dit.safetensors")
            if os.path.exists(local_dit):
                self.ckpt_path = local_dit
            elif os.path.exists(os.path.join(base_dir, "ckpts/AuK-Flash/auk_flash.safetensors")):
                self.ckpt_path = os.path.join(base_dir, "ckpts/AuK-Flash/auk_flash.safetensors")
            else:
                from huggingface_hub import snapshot_download
                print(f"Downloading MLX model from Hugging Face ({repo_id})...")
                snapshot_download(repo_id=repo_id, local_dir=mlx_dir)
                self.ckpt_path = local_dit
        else:
            self.ckpt_path = ckpt_path

        # 2. Resolve VAE path
        if vae_path is None:
            local_vae = os.path.join(mlx_dir, "vae.safetensors")
            if os.path.exists(local_vae):
                self.vae_path = local_vae
            elif os.path.exists(os.path.join(base_dir, "ckpts/AuK-Flash/vae.safetensors")):
                self.vae_path = os.path.join(base_dir, "ckpts/AuK-Flash/vae.safetensors")
            else:
                self.vae_path = local_vae
        else:
            self.vae_path = vae_path

        # 3. Resolve Config path
        if config_path is None:
            local_cfg = os.path.join(mlx_dir, "config.json")
            if os.path.exists(local_cfg):
                self.config_path = local_cfg
            elif os.path.exists(os.path.join(base_dir, "ckpts/AuK-Flash/config.yaml")):
                self.config_path = os.path.join(base_dir, "ckpts/AuK-Flash/config.yaml")
            else:
                self.config_path = local_cfg
        else:
            self.config_path = config_path

        # 4. Resolve Qwen feature encoder
        if qwen_path is None:
            local_qwen = os.path.join(base_dir, "ckpts/Qwen2.5-Omni-3B")
            if os.path.exists(local_qwen):
                self.qwen_path = local_qwen
            else:
                self.qwen_path = "Qwen/Qwen2.5-Omni-3B"
        else:
            self.qwen_path = qwen_path

        self.device = device
        self.target_sample_rate = 24000

        print(f"Loading full-parity AuK inference engine on {self.device}...")
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
        spoken_text: Optional[str] = None,
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

        ref_duration = 0.0
        if audio_path and os.path.exists(audio_path):
            wav_np, sr = sf.read(audio_path)
            if wav_np.ndim == 2:
                wav_np = wav_np.mean(axis=-1)
            ref_duration = len(wav_np) / float(sr)
            t_audio = torch.from_numpy(wav_np).float().unsqueeze(0)
            audio_arg = (t_audio, sr)
        elif isinstance(audio, tuple):
            audio_arg = audio
            ref_duration = audio[0].shape[-1] / float(audio[1])

        # Resolve duration intelligently
        final_gen_seconds = gen_seconds
        if final_gen_seconds is None or final_gen_seconds <= 0:
            if ref_duration > 0 and not spoken_text:
                # Editing tasks preserve exact input duration
                final_gen_seconds = ref_duration
            elif spoken_text:
                final_gen_seconds = estimate_spoken_duration(spoken_text)
            else:
                # Try extracting spoken content from messages
                extracted_text = ""
                for m in messages:
                    if m.get("role") == "user":
                        for c in m.get("content", []):
                            if isinstance(c, dict) and c.get("type") == "text":
                                txt = c.get("text", "")
                                if 'generate speech content "' in txt:
                                    extracted_text = txt.split('generate speech content "')[-1].rstrip('". ')
                                elif 'Say the following with the same voice: "' in txt:
                                    extracted_text = txt.split('Say the following with the same voice: "')[-1].rstrip('". ')
                                elif '，生成语音内容"' in txt:
                                    extracted_text = txt.split('，生成语音内容"')[-1].rstrip('". ')
                if extracted_text:
                    final_gen_seconds = estimate_spoken_duration(extracted_text)
                elif ref_duration > 0:
                    final_gen_seconds = ref_duration
                else:
                    final_gen_seconds = 3.5

        # Contextual prompt anchoring for edit tasks
        adapted_messages = []
        for m in messages:
            new_m = dict(m)
            if m.get("role") == "user":
                new_content = []
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        orig_t = c.get("text", "")
                        adapted_t = adapt_edit_instruction(orig_t, spoken_text=spoken_text)
                        new_content.append({**c, "text": adapted_t})
                    else:
                        new_content.append(c)
                new_m["content"] = new_content
            adapted_messages.append(new_m)

        wav, sr = self.engine.generate(
            adapted_messages,

            audio=audio_arg,
            gen_seconds=final_gen_seconds,
            seed=seed,
        )

        total_latency = time.perf_counter() - t_start
        wav_np = wav.squeeze().cpu().numpy()
        actual_duration = len(wav_np) / float(sr)
        rtf = total_latency / actual_duration if actual_duration > 0 else 0.0

        metrics = {
            "latency": total_latency,
            "audio_duration": actual_duration,
            "target_seconds": final_gen_seconds,
            "rtf": rtf,
            "sample_rate": sr,
        }
        return wav_np, sr, metrics


def save_audio(waveform: np.ndarray, sample_rate: int, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, waveform, sample_rate)
