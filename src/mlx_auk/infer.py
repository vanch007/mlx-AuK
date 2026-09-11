import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import soundfile as sf

from .config import AuKConfig, BigVGANConfig, Flux2EditConfig
from .dit.cfm import CFMEdit
from .dit.flux2 import Flux2Edit
from .thinker.encoder import QwenOmniConditionEncoder
from .vae.vae import BigVGANFlowVAE


class AukInfer:
    def __init__(
        self,
        config: Optional[AuKConfig] = None,
        model_dir: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        qwen_path: Optional[str] = None,
    ):
        self.config = config or AuKConfig.auk_flash()
        if qwen_path:
            self.config.text_encoder_path = os.path.abspath(qwen_path)

        self.target_sample_rate = self.config.vae.target_sample_rate
        self.downsample_rate = self.config.vae.downsample_rate
        self.latent_dim = self.config.vae.latent_dim
        self.is_flash = self.config.is_flash

        self.vae = BigVGANFlowVAE(self.config.vae)
        self.transformer = Flux2Edit(self.config.dit)
        self.cfm = CFMEdit(self.transformer, num_channels=self.latent_dim)
        self.thinker = QwenOmniConditionEncoder(self.config.text_encoder_path)

        if model_dir and os.path.isdir(model_dir):
            dit_file = os.path.join(model_dir, "dit.safetensors")
            vae_file = os.path.join(model_dir, "vae.safetensors")
            self.load_mlx_weights(dit_file, vae_file)
        elif ckpt_path:
            self.load_weights(ckpt_path, vae_path)

    def load_mlx_weights(self, dit_path: str, vae_path: Optional[str] = None):
        print("Loading native MLX weights directly via mx.load()...")
        if os.path.exists(dit_path):
            weights = mx.load(dit_path)
            print("Loaded %d native DiT tensors directly into MLX!" % len(weights))
            if "layer_weights" in weights:
                self.cfm.layer_weights = weights["layer_weights"]
            if "layer_scale" in weights:
                self.cfm.layer_scale = float(np.array(weights["layer_scale"]).item())

        if vae_path and os.path.exists(vae_path):
            vae_w = mx.load(vae_path)
            print("Loaded %d native VAE tensors directly into MLX!" % len(vae_w))
            if "global_mean" in vae_w:
                self.vae.global_mean = vae_w["global_mean"]
            if "global_log_std" in vae_w:
                self.vae.global_log_std = vae_w["global_log_std"]

    def load_weights(self, ckpt_path: str, vae_path: Optional[str] = None):
        print("Loading weights into MLX AuK from %s..." % ckpt_path)
        weights = {}
        if os.path.exists(ckpt_path):
            try:
                weights = mx.load(ckpt_path)
                print("Loaded %d weight tensors directly into MLX!" % len(weights))
            except Exception:
                from safetensors.torch import load_file
                st = load_file(ckpt_path)
                for k, v in st.items():
                    clean_k = k.replace("ema_model.", "")
                    if not clean_k.startswith("text_encoder."):
                        weights[clean_k] = mx.array(v.float().numpy())
                print("Converted and loaded %d non-text-encoder tensors into MLX!" % len(weights))

        if "layer_weights" in weights:
            self.cfm.layer_weights = weights["layer_weights"]
        if "layer_scale" in weights:
            self.cfm.layer_scale = float(np.array(weights["layer_scale"]).item())

        if vae_path and os.path.exists(vae_path):
            from safetensors.torch import load_file
            st_vae = load_file(vae_path)
            vae_w = {k: mx.array(v.float().numpy()) for k, v in st_vae.items()}
            print("Loaded %d VAE tensors into MLX!" % len(vae_w))
            if "global_mean" in vae_w:
                self.vae.global_mean = vae_w["global_mean"]
            if "global_log_std" in vae_w:
                self.vae.global_log_std = vae_w["global_log_std"]

    def load_audio(self, source: str) -> Tuple[mx.array, float]:
        import librosa
        wav, sr = librosa.load(source, sr=self.target_sample_rate, mono=True)
        rms = float(np.sqrt(np.mean(wav**2)))
        arr = mx.array(wav)[None, None, :]
        return arr, rms

    def generate(
        self,
        messages: List[Dict[str, any]],
        *,
        audio: Optional[str] = None,
        gen_seconds: Optional[float] = None,
        nfe: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, Dict[str, float]]:
        t_start = time.perf_counter()

        ref_audio = None
        ref_rms = None
        audio_path = audio

        if audio_path is None:
            for m in messages:
                if m.get("role") == "user":
                    for c in m.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "audio":
                            audio_path = c.get("audio") or c.get("audio_url")
                            break

        if audio_path and os.path.exists(audio_path):
            ref_audio, ref_rms = self.load_audio(audio_path)
            ref_len_samples = ref_audio.shape[-1]
            ref_latent_len = ref_len_samples // self.downsample_rate
            ref_latents = self.vae.encoding_and_normalization(ref_audio)
        else:
            ref_latent_len = 0
            ref_latents = mx.zeros((1, 0, self.latent_dim))
            for m in messages:
                if m.get("role") == "user":
                    for c in m.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "text" and not c["text"].endswith("|<no_prompt_audio>|"):
                            c["text"] = c["text"] + "|<no_prompt_audio>|"

        if gen_seconds is not None:
            gen_latent_len = max(1, int(math.ceil(gen_seconds * self.target_sample_rate / self.downsample_rate)))
        else:
            gen_latent_len = max(1, ref_latent_len if ref_latent_len > 0 else int(3.0 * self.target_sample_rate / self.downsample_rate))

        t_encode_start = time.perf_counter()
        text_embed = self.thinker.encode(messages, self.cfm.layer_weights, self.cfm.layer_scale)
        mx.eval(text_embed)
        t_encode = time.perf_counter() - t_encode_start

        if self.is_flash:
            steps = 4
            cfg = 0.0
            t_grid = self.config.flash_t_grid
        else:
            steps = nfe or self.config.default_nfe
            cfg = cfg_strength or self.config.default_cfg
            t_grid = None

        t_sample_start = time.perf_counter()
        _, gen_latent = self.cfm.sample(
            cond=ref_latents,
            text_embed=text_embed,
            target_len=gen_latent_len,
            steps=steps,
            cfg_strength=cfg,
            t_grid=t_grid,
            seed=seed,
        )
        mx.eval(gen_latent)
        t_sample = time.perf_counter() - t_sample_start

        t_decode_start = time.perf_counter()
        denorm_latent = self.vae.denormalize(gen_latent)
        latent_bct = mx.transpose(denorm_latent, (0, 2, 1))
        audio_out = self.vae.inference_from_latents(latent_bct)
        mx.eval(audio_out)
        t_decode = time.perf_counter() - t_decode_start

        total_latency = time.perf_counter() - t_start
        wav_np = np.array(audio_out.squeeze())
        audio_duration = len(wav_np) / float(self.target_sample_rate)
        rtf = total_latency / audio_duration if audio_duration > 0 else 0.0

        metrics = {
            "latency": total_latency,
            "encode_time": t_encode,
            "sample_time": t_sample,
            "decode_time": t_decode,
            "audio_duration": audio_duration,
            "rtf": rtf,
        }
        return wav_np, self.target_sample_rate, metrics

def save_audio(waveform: np.ndarray, sample_rate: int, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, waveform, sample_rate)
