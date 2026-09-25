"""Qwen3-VL VLA bridge — loads a frozen HF checkpoint (read-only).

Default weights (do not copy/modify under redacted/):
  .../redacted/.../qwen3-vl-8b-VPT-gui-with-aux-merged-weighted/.../checkpoint-16200

Aligned with OpenHA cold-start eval protocol:
  - system prompt = MINECRAFT_V1 (full)
  - message_mode = multi_turn with history_window (default 10)
  - user text = Task:… / Choose your next actions.

Modes
-----
  hf     : transformers generate (needs GPU + enough memory)
  stub   : deterministic noop / forward for plumbing tests
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from curriculum.action_codec import (
    PROMPT_MINECRAFT_LEGACY,
    PROMPT_MINECRAFT_V1,
    chunk_to_mg2,
    cold_start_step_instruction,
    noop_action,
    parse_actions_text,
)
from curriculum.policy import ActResult, PolicyBase


def _images_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """Collect PIL/image payloads in the same order the chat template emits pads."""
    out: List[Any] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" or item.get("image") is not None:
                img = item.get("image")
                if img is not None:
                    out.append(img)
    return out


# base VLA (read-only path)
DEFAULT_VLA_CKPT = "/path/to/vla_checkpoint"


@dataclass
class QwenVLAConfig:
    model_path: str = DEFAULT_VLA_CKPT
    mode: str = "hf"  # hf | stub
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 1.0
    do_sample: bool = True
    action_chunks_len: int = 4
    instruction: str = "Explore and progress the Minecraft task."
    history_window: int = 10  # OpenHA / redacted default
    protocol: str = "minecraft_v1"  # minecraft_v1 | legacy_k1_t0


class QwenVLAPolicy(PolicyBase):
    """Vision-language-action policy emitting MineStudio action chunks + pretrained_wm tensors."""

    def __init__(self, cfg: Optional[QwenVLAConfig] = None):
        super().__init__()
        self.cfg = cfg or QwenVLAConfig()
        self.model = None
        self.processor = None
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.cfg.mode == "hf":
            self._load_hf()

    def _load_hf(self):
        path = self.cfg.model_path
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"VLA checkpoint not found (read-only expected path): {path}"
            )
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        dtype = getattr(torch, self.cfg.dtype)
        try:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                path, torch_dtype=dtype, trust_remote_code=True, device_map=self.cfg.device,
            )
        except Exception:
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    path, torch_dtype=dtype, trust_remote_code=True, device_map=self.cfg.device,
                )
            except Exception as e:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    path, torch_dtype=dtype, trust_remote_code=True, device_map=self.cfg.device,
                )
                print(f"[QwenVLA] loaded via AutoModel fallback: {e}")
        self.model.eval()
        print(f"[QwenVLA] loaded {path} mode=hf protocol={self.cfg.protocol}")

    def _build_messages(
        self,
        image: Image.Image,
        instruction: str,
        *,
        step_num: int,
        chat_history: Optional[List[Dict[str, Any]]],
        history_window: int,
        extra_user_text: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Build either exact legacy k1_t0 or OpenHA minecraft_v1 messages."""
        protocol = str(self.cfg.protocol or "minecraft_v1").lower()
        if protocol in ("legacy", "legacy_k1_t0", "k1_t0", "single_turn"):
            sys_prompt = PROMPT_MINECRAFT_LEGACY.format(instruction=instruction)
            user_text = (
                f"Step instruction: {instruction}\n"
                "Output the next <actions> chunk."
            )
            if extra_user_text:
                user_text += "\nRecent actions:\n" + str(extra_user_text)
            return [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sys_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ],
                },
            ]

        sys_prompt = PROMPT_MINECRAFT_V1.format(instruction=instruction)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        ]
        hist = list(chat_history or [])
        if history_window > 0:
            hist = hist[-int(history_window) :]
        for rec in hist:
            hist_img = rec.get("image")
            if hist_img is None:
                continue
            hist_instr = cold_start_step_instruction(
                instruction, int(rec.get("step_num", 0))
            )
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": hist_img},
                    {"type": "text", "text": hist_instr},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": str(rec.get("assistant_content") or ""),
            })

        cur_instr = cold_start_step_instruction(instruction, int(step_num))
        user_content: List[Dict[str, Any]] = [
            {"type": "image", "image": image},
            {"type": "text", "text": cur_instr},
        ]
        if extra_user_text:
            user_content.append({"type": "text", "text": str(extra_user_text)})
        messages.append({"role": "user", "content": user_content})
        return messages

    @torch.no_grad()
    def act_image(
        self,
        image: Image.Image,
        instruction: Optional[str] = None,
        history_text: Optional[List[str]] = None,
        *,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        step_num: int = 0,
        history_window: Optional[int] = None,
        extra_user_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          actions: List[MineStudio dict] (len≈4)
          text: raw generation
          keyboard, mouse: pretrained_wm tensors [T,4]/[T,2]
          done: bool
        """
        instr = instruction or self.cfg.instruction
        if self.cfg.mode == "stub":
            acts = [noop_action() for _ in range(self.cfg.action_chunks_len)]
            for a in acts:
                a["forward"] = 1
            kb, ms = chunk_to_mg2(acts)
            return {
                "actions": acts,
                "text": "<actions> keyPress(w) ; keyPress(w) ; keyPress(w) ; keyPress(w) </actions>",
                "keyboard": kb, "mouse": ms, "done": False,
            }

        assert self.model is not None and self.processor is not None
        # Legacy: history_text → append onto current user turn (memory tips etc.)
        extra = extra_user_text
        if history_text:
            tip = "Recent context:\n" + "\n".join(history_text[-5:])
            extra = (extra + "\n" + tip) if extra else tip

        win = int(self.cfg.history_window if history_window is None else history_window)
        messages = self._build_messages(
            image,
            instr,
            step_num=int(step_num),
            chat_history=chat_history,
            history_window=win,
            extra_user_text=extra,
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs: Optional[List[Any]] = None
        video_inputs = None
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            image_inputs = None
            video_inputs = None
        if not image_inputs:
            image_inputs = _images_from_messages(messages)
        if not image_inputs:
            image_inputs = [image]
        n_pad = text.count("<|image_pad|>")
        n_img = len(image_inputs)
        if n_pad != n_img:
            raise RuntimeError(
                f"QwenVLA image/token mismatch: image_pad={n_pad} images={n_img} "
                f"(history={len(chat_history or [])} window={win})"
            )
        proc_kwargs: Dict[str, Any] = {
            "text": [text],
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
        inputs = self.processor(**proc_kwargs)

        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.cfg.max_new_tokens),
            "do_sample": bool(self.cfg.do_sample),
        }
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = float(self.cfg.temperature)
        gen = self.model.generate(**inputs, **gen_kwargs)
        in_len = inputs["input_ids"].shape[-1]
        out_ids = gen[:, in_len:]
        text_out = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        actions, done = parse_actions_text(text_out, self.cfg.action_chunks_len)
        if not actions:
            actions = [noop_action() for _ in range(self.cfg.action_chunks_len)]
        kb, ms = chunk_to_mg2(actions)
        return {
            "actions": actions, "text": text_out,
            "keyboard": kb, "mouse": ms, "done": done,
        }

    def act(self, latent: torch.Tensor, deterministic: bool = False) -> ActResult:
        """PolicyBase compatibility — latent-only path is unsupported for VLA."""
        B = latent.shape[0] if latent.ndim >= 4 else 1
        device = self._dummy.device
        kb = torch.zeros(B, 4, device=device)
        ms = torch.zeros(B, 2, device=device)
        z = torch.zeros(B, device=device)
        return ActResult(keyboard=kb, mouse=ms, log_prob=z, value=z, entropy=z)
