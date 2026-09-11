import os
import torch
import torch.nn.functional as F
import mlx.core as mx
from typing import List, Dict, Any, Optional

class QwenOmniConditionEncoder:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = os.path.abspath(model_path)
        self._loaded = False
        self.processor = None
        self.thinker = None

    def load(self):
        if self._loaded:
            return
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        
        print(f"Loading Qwen2.5-Omni processor and thinker from {self.model_path}...")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if hasattr(self.thinker, "visual") and self.thinker.visual is not None:
            del self.thinker.visual
            self.thinker.visual = None
        self.thinker = self.thinker.to(self.device).eval()
        self.thinker.requires_grad_(False)
        self._loaded = True

    def encode(
        self,
        messages: List[Dict[str, Any]],
        layer_weights: mx.array,
        layer_scale: float = 1.0,
    ) -> mx.array:
        self.load()
        from qwen_omni_utils import process_mm_info

        formatted_text = self.processor.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        has_mm = any(
            isinstance(c, dict) and "type" in c and c["type"] in ("audio", "image", "video")
            for m in messages
            for c in m.get("content", [])
        )

        if has_mm:
            mm_audios, mm_images, mm_videos = process_mm_info([messages], use_audio_in_video=True)
            inputs = self.processor(
                text=formatted_text,
                audio=mm_audios,
                images=mm_images,
                videos=mm_videos,
                padding=True,
                return_tensors="pt",
                use_audio_in_video=True,
            )
        else:
            inputs = self.processor(
                text=formatted_text,
                padding=True,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.thinker(**inputs, output_hidden_states=True)

        all_hidden_states = outputs.hidden_states # tuple of (num_layers + 1)
        d_llm = all_hidden_states[0].shape[-1]

        stacked = torch.stack([F.layer_norm(h, [d_llm]) for h in all_hidden_states[1:]], dim=0)
        
        # layer_weights is MLX array, convert to torch tensor for fusion
        w_np = layer_weights.astype(mx.float32)
        import numpy as np
        w_th = torch.from_numpy(np.array(w_np)).to(self.device)
        weights = F.softmax(w_th, dim=0)

        hidden = (stacked * weights[:, None, None, None]).sum(dim=0) * layer_scale
        hidden_np = hidden.float().cpu().numpy()
        return mx.array(hidden_np)
