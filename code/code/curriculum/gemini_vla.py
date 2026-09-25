"""Gemini Flash VLA via an OpenAI-compatible LLM gateway.

Endpoint (override with LLM_API_REQUEST_URL):
  https://api.example.com/v1/chat/completions

Auth:
  Authorization: Bearer {API_KEY}

Env (first match wins for API key):
  LLM_API_APP_ID / GEMINI_API_KEY / GOOGLE_API_KEY
  GEMINI_MODEL   (default: gemini-2.5-flash)

Same action contract as Qwen cold-start:
  <actions> keyPress(w) ; ... </actions>
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from curriculum.action_codec import (
    PROMPT_MINECRAFT_NO_THOUGHT,
    chunk_to_mg2,
    noop_action,
    parse_actions_text,
)
from curriculum.policy import ActResult, PolicyBase


DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_REQUEST_URL = os.environ.get(
    "LLM_API_REQUEST_URL",
    "https://api.example.com/v1/chat/completions",
)
# No built-in app id. Set LLM_API_APP_ID, GEMINI_API_KEY, or GOOGLE_API_KEY.
_DEFAULT_APP_ID = ""


@dataclass
class GeminiVLAConfig:
    model: str = DEFAULT_GEMINI_MODEL
    api_key: str = ""  # LLM API key (Bearer token)
    request_url: str = DEFAULT_REQUEST_URL
    mode: str = "api"  # api | stub
    max_output_tokens: int = 512
    temperature: float = 0.7
    action_chunks_len: int = 4
    instruction: str = "Explore and progress the Minecraft task."
    timeout_s: float = 90.0
    max_retries: int = 8
    retry_base_s: float = 3.0
    min_interval_s: float = 2.0  # RPM guard for API quota


def _resolve_app_id(explicit: str = "") -> str:
    return (
        (explicit or "").strip()
        or os.environ.get("LLM_API_APP_ID", "").strip()
        or os.environ.get("LLM_API_APP_ID", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
        or _DEFAULT_APP_ID
    )


class GeminiVLAPolicy(PolicyBase):
    """Multimodal Gemini Flash (LLM API gateway) → MineStudio action chunks."""

    def __init__(self, cfg: Optional[GeminiVLAConfig] = None):
        super().__init__()
        self.cfg = cfg or GeminiVLAConfig()
        self.cfg.api_key = _resolve_app_id(self.cfg.api_key)
        self.cfg.request_url = (
            (self.cfg.request_url or "").strip() or DEFAULT_REQUEST_URL
        )
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self._last_call_ts = 0.0
        if self.cfg.mode == "api" and not self.cfg.api_key:
            raise RuntimeError(
                "Gemini VLA needs LLM_API_APP_ID / LLM_API_APP_ID "
                "(or GeminiVLAConfig.api_key)."
            )
        print(
            f"[GeminiVLA] llm_api model={self.cfg.model} mode={self.cfg.mode} "
            f"app_id={'set' if self.cfg.api_key else 'missing'}",
            flush=True,
        )

    def _image_b64_jpeg(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        # OpenAI-style
        choices = payload.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(str(p.get("text") or ""))
                    elif isinstance(p, str):
                        parts.append(p)
                joined = "\n".join(x for x in parts if x).strip()
                if joined:
                    return joined
        # Some API wrappers put text at top-level
        top = payload.get("content")
        if isinstance(top, str) and top.strip():
            return top.strip()
        raise RuntimeError(f"LLM API empty content: {json.dumps(payload)[:500]}")

    def _generate(self, image: Image.Image, user_text: str, system: str) -> str:
        b64 = self._image_b64_jpeg(image)
        body = {
            "model": self.cfg.model,
            "stream": False,
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_output_tokens),
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                    ],
                },
            ],
        }
        data = json.dumps(body).encode("utf-8")
        last_err = ""
        for attempt in range(int(self.cfg.max_retries) + 1):
            gap = float(self.cfg.min_interval_s)
            if gap > 0 and self._last_call_ts > 0:
                wait = gap - (time.time() - self._last_call_ts)
                if wait > 0:
                    time.sleep(wait)
            req = urllib.request.Request(
                self.cfg.request_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.cfg.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=float(self.cfg.timeout_s)) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self._last_call_ts = time.time()
                return self._extract_text(payload)
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", errors="replace")
                last_err = f"LLM API HTTP {e.code}: {err[:500]}"
                retryable = e.code in (429, 500, 502, 503, 504) or (
                    "RESOURCE_EXHAUSTED" in err or "rate" in err.lower()
                )
                if (not retryable) or attempt >= int(self.cfg.max_retries):
                    raise RuntimeError(last_err) from e
                sleep_s = float(self.cfg.retry_base_s) * (2 ** attempt)
                print(
                    f"[GeminiVLA] retry {attempt + 1}/{self.cfg.max_retries} "
                    f"after HTTP {e.code}; sleep {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            except Exception as e:
                last_err = f"LLM API request failed: {e}"
                if attempt >= int(self.cfg.max_retries):
                    raise RuntimeError(last_err) from e
                sleep_s = float(self.cfg.retry_base_s) * (2 ** attempt)
                print(
                    f"[GeminiVLA] retry {attempt + 1}/{self.cfg.max_retries} "
                    f"after {type(e).__name__}; sleep {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
        raise RuntimeError(last_err or "LLM API request failed")

    @torch.no_grad()
    def act_image(
        self,
        image: Image.Image,
        instruction: Optional[str] = None,
        history_text: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        instr = instruction or self.cfg.instruction
        if self.cfg.mode == "stub":
            acts = [noop_action() for _ in range(self.cfg.action_chunks_len)]
            for a in acts:
                a["forward"] = 1
            kb, ms = chunk_to_mg2(acts)
            return {
                "actions": acts,
                "text": "<actions> keyPress(w) ; keyPress(w) ; keyPress(w) ; keyPress(w) </actions>",
                "keyboard": kb,
                "mouse": ms,
                "done": False,
            }

        from curriculum.action_codec import PROMPT_MINECRAFT_V1, cold_start_step_instruction

        system = PROMPT_MINECRAFT_V1.format(instruction=instr)
        step_num = int(kwargs.get("step_num", 0) or 0)
        user = cold_start_step_instruction(instr, step_num)
        extra = kwargs.get("extra_user_text")
        if extra:
            user = user + "\n" + str(extra)
        if history_text:
            user += "\n\nRecent context:\n" + "\n".join(history_text[-5:])
        text_out = self._generate(image, user, system)
        actions, done = parse_actions_text(text_out, self.cfg.action_chunks_len)
        if not actions:
            actions = [noop_action() for _ in range(self.cfg.action_chunks_len)]
        kb, ms = chunk_to_mg2(actions)
        return {
            "actions": actions,
            "text": text_out,
            "keyboard": kb,
            "mouse": ms,
            "done": done,
        }

    def act(self, latent: torch.Tensor, deterministic: bool = False) -> ActResult:
        B = latent.shape[0] if latent.ndim >= 4 else 1
        device = self._dummy.device
        kb = torch.zeros(B, 4, device=device)
        ms = torch.zeros(B, 2, device=device)
        z = torch.zeros(B, device=device)
        return ActResult(keyboard=kb, mouse=ms, log_prob=z, value=z, entropy=z)


def ping_gemini(image: Optional[Image.Image] = None) -> Dict[str, Any]:
    """Minimal connectivity check via LLM API gateway."""
    img = image or Image.new("RGB", (64, 64), color=(80, 140, 80))
    pol = GeminiVLAPolicy(GeminiVLAConfig(instruction="Move forward toward the tree."))
    out = pol.act_image(img)
    return {
        "ok": True,
        "model": pol.cfg.model,
        "request_url": pol.cfg.request_url,
        "text": out.get("text", "")[:400],
        "n_actions": len(out.get("actions") or []),
    }
