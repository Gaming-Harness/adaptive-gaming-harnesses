"""LLM harness proposer: LLMAPI API invents structured harness patches.

Frozen VLA. LLM never emits game actions. Sandbox remains the sole judge.

Meaningful patch = human-readable diagnosis + why + proposal for the VLA,
implemented mainly via prompt/instruction edits (not knob-only retries).

Patch schema::

    {
      "diagnosis": "what the agent sees / where it is stuck",
      "why": "why that causes failure",
      "proposal": "what Qwen-VLA should do instead (behavioral guidance)",
      "edits": [
        {"path": ["recovery_prompt"], "value": "..."},
        {"path": ["runtime", "instruction"], "value": "..."}
      ],
      "claim": "short",
      "expect_delta": 0.05
    }

Knob-only patches (retry/budget/pool without a prompt edit) are rejected.
Proposer failure (truncated JSON / no edits) is NOT harness failure.
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

from curriculum.harness_schema import (
    HarnessSpec,
    MemorySpec,
    ProbeSpec,
    RecoverSpec,
    VerifySpec,
)
from curriculum.sandbox_experience import ACTION_NAMES


FROZEN_RUNTIME = {
    "backend", "vla_mode", "vla_model_path", "vla_device", "vla_dtype",
    "task_config", "img_save_dir", "gemini_api_key", "vla_protocol",
}

ALLOWED_TOP = {"task_prompt", "decompose_prompt", "recovery_prompt"}
ALLOWED_NESTED = {
    "probe": set(ProbeSpec.__dataclass_fields__),
    "memory": set(MemorySpec.__dataclass_fields__),
    "verify": set(VerifySpec.__dataclass_fields__),
    "recover": set(RecoverSpec.__dataclass_fields__),
    "runtime": {
        "max_steps", "action_chunk_len", "checkpoint_every",
        "instruction", "ticks_per_action", "success_reward_thresh",
        "soft_rollback",
    },
}

# Edits that change what the frozen VLA is told to do.
PROMPT_EDIT_PATHS = {
    ("task_prompt",),
    ("decompose_prompt",),
    ("recovery_prompt",),
    ("runtime", "instruction"),
}

SYSTEM = (
    "You are a Harness researcher for a frozen Minecraft VLA (Qwen). "
    "You do NOT output low-level game actions (w/a/s/d/attack/keyPress) as "
    "standalone content. However, you MAY include a <actions>...</actions> "
    "block as plain-text inside prompt/instruction edits (e.g. "
    "recovery_prompt/runtime.instruction) so the frozen VLA can follow it. "
    "Your job is human-understandable: (1) diagnose what is wrong in the POV, "
    "(2) explain why, (3) propose how Qwen should behave, "
    "(4) encode that proposal into prompt/instruction edits. "
    "When RESEARCH EVIDENCE (failure clusters + attribution probes) is provided, "
    "GROUND diagnosis/proposal in that evidence; do not ignore ruled-out hypotheses. "
    "SUCCESS MEMORY is transferable skills with a {target} placeholder. "
    "Instantiate the matching skill for THIS task; never copy another task's "
    "block/entity name or a fixed mouseMove(dx,dy). "
    "Do NOT propose knob-only patches (max_retries / budget / action_pool alone). "
    "Knobs are optional secondary edits only after a prompt/instruction change. "
    "If you edit `recovery_prompt`, include exactly ONE minimal <actions>...</actions> "
    "block inside the recovery_prompt text, using only: mouseMove, mouseClick, "
    "keyPress, no_op. Keep it short (4 steps total; separated by ';'). "
    "JSON fields: diagnosis, why, proposal, edits, claim, expect_delta, "
    "optional predicted_regression. "
    "diagnosis/why/proposal must be concrete. No markdown. No essays."
)

# Extra constraints for the mtv1_actionable experiment. Default proposer stays
# unchanged unless propose_patch(..., actionable_patches=True).
SYSTEM_ACTIONABLE = (
    " OpenHA tasks are assumed executable from their initialized scene "
    "(teleport, tools, world seed). A screenshot is PARTIAL evidence, not proof "
    "the task is impossible. NEVER tell Qwen to stop, give up, report inability, "
    "or skip mining/killing/crafting. Qwen can only emit <actions>, so "
    "proposals must be executable behavior: look/scan, reorient, approach, "
    "center the target, attack/use, enter portal if needed. "
    "If a previous proposal was REVERTED, do not repeat that diagnosis or strategy."
)

ABANDON_PATTERNS = (
    r"cannot complete",
    r"can't complete",
    r"can not complete",
    r"report(?: that you)? (?:cannot|inability|unable)",
    r"report inability",
    r"do not attempt",
    r"don't attempt",
    r"cease attempts",
    r"stop attempting",
    r"stop\b.{0,40}\b(?:task|mining|mine)",
    r"task cannot be completed",
    r"unable to complete",
    r"netherrack cannot be found here",
    r"do not mine",
    r"don't mine",
    r"give up",
    r"impossible (?:here|in this)",
)

ACTIONABLE_CUES = (
    "look down", "look around", "look up", "scan", "aim", "center",
    "approach", "attack", "mine", "enter", "portal", "activate",
    "reorient", "find the", "search", "turn", "go to", "break",
    "chop", "kill", "craft", "place", "use the",
)

TRUNC_REASONS = {"length", "max_tokens", "max_output_tokens", "MAX_TOKENS"}


def _path_tuple(path: Any) -> Tuple[str, ...]:
    if isinstance(path, (list, tuple)):
        return tuple(str(x) for x in path)
    return (str(path),)


def is_prompt_edit_path(path: Any) -> bool:
    t = _path_tuple(path)
    if t in PROMPT_EDIT_PATHS:
        return True
    # allow ["runtime","instruction"] already covered; also bare instruction under runtime
    return len(t) >= 2 and t[0] == "runtime" and t[1] == "instruction"


def patch_has_prompt_edit(patch: Dict[str, Any]) -> bool:
    for ed in patch.get("edits") or []:
        if isinstance(ed, dict) and is_prompt_edit_path(ed.get("path")):
            return True
    return False


def normalize_meaningful_fields(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Fill diagnosis/why/proposal; keep backward compat with claim/reason."""
    out = dict(patch)
    diagnosis = str(out.get("diagnosis") or "").strip()
    why = str(out.get("why") or "").strip()
    proposal = str(out.get("proposal") or "").strip()
    claim = str(out.get("claim") or "").strip()
    reason = str(out.get("reason") or "").strip()
    if not diagnosis and reason:
        diagnosis = reason
    if not why and claim:
        why = claim
    if not proposal:
        proposal = claim or reason
    out["diagnosis"] = diagnosis[:240]
    out["why"] = why[:240]
    out["proposal"] = proposal[:320]
    if claim:
        out["claim"] = claim[:120]
    elif proposal:
        out["claim"] = proposal[:120]
    if reason:
        out["reason"] = reason[:80]
    return out


def validate_meaningful_patch(patch: Dict[str, Any]) -> None:
    """Reject knob-only / missing-diagnosis patches."""
    p = normalize_meaningful_fields(patch)
    if len(str(p.get("diagnosis") or "").strip()) < 8:
        raise ValueError("missing diagnosis (what is wrong in POV / behavior)")
    if len(str(p.get("why") or "").strip()) < 8:
        raise ValueError("missing why (why that causes failure)")
    if len(str(p.get("proposal") or "").strip()) < 8:
        raise ValueError("missing proposal (how Qwen should act)")
    if not (p.get("edits") or []):
        raise ValueError("patch missing edits[]")
    if not patch_has_prompt_edit(p):
        raise ValueError(
            "edits must include a prompt/instruction change "
            "(task_prompt / recovery_prompt / runtime.instruction); "
            "knob-only patches are not meaningful"
        )


def _patch_text_blob(patch: Dict[str, Any]) -> str:
    parts = [
        str(patch.get("diagnosis") or ""),
        str(patch.get("why") or ""),
        str(patch.get("proposal") or ""),
        str(patch.get("claim") or ""),
    ]
    for ed in patch.get("edits") or []:
        if isinstance(ed, dict):
            parts.append(str(ed.get("value") or ""))
    return " ".join(parts).lower()


def validate_actionable_patch(patch: Dict[str, Any]) -> None:
    """Reject give-up / impossible-task patches; require executable behavior."""
    p = normalize_meaningful_fields(patch)
    blob = _patch_text_blob(p)
    for pat in ABANDON_PATTERNS:
        if re.search(pat, blob, flags=re.I):
            raise ValueError(
                "abandon/impossible patch rejected: OpenHA tasks are assumed "
                "executable; do not stop, report inability, or skip the goal. "
                f"matched={pat!r}"
            )
    if not any(cue in blob for cue in ACTIONABLE_CUES):
        raise ValueError(
            "proposal/edits must include executable behavior "
            "(look/scan/aim/center/approach/attack/mine/enter portal/...)"
        )


def _looks_like_patch(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("edits"):
        return True
    if obj.get("component") and obj.get("op"):
        return True
    if "claim" in obj and ("edits" in obj or "expect_delta" in obj):
        return True
    return False


def extract_json_obj(text: str) -> Dict[str, Any]:
    """Parse first JSON object from messy LLMAPI / markdown output."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    decoder = json.JSONDecoder()
    idx = raw.find("{")
    last_err: Optional[Exception] = None
    fallback: Optional[Dict[str, Any]] = None
    while idx >= 0:
        try:
            obj, _end = decoder.raw_decode(raw[idx:])
            if isinstance(obj, dict):
                if _looks_like_patch(obj):
                    return obj
                if fallback is None:
                    fallback = obj
        except json.JSONDecodeError as e:
            last_err = e
        idx = raw.find("{", idx + 1)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and _looks_like_patch(obj):
                return obj
            if isinstance(obj, dict) and fallback is None:
                fallback = obj
        except json.JSONDecodeError as e:
            last_err = e
    if fallback is not None:
        return fallback
    raise ValueError(f"no JSON object in LLM output ({last_err}): {raw[:400]!r}")


def _close_truncated_json(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    if "{" in t and not t.lstrip().startswith("{"):
        t = t[t.find("{") :]
    in_str = False
    escape = False
    stack: List[str] = []
    for ch in t:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        t += '"'
    # drop trailing comma before closing
    t = re.sub(r",\s*$", "", t)
    t += "".join(reversed(stack))
    return t


def _extract_edits_array(text: str) -> Optional[List[Any]]:
    idx = (text or "").find('"edits"')
    if idx < 0:
        return None
    bracket = text.find("[", idx)
    if bracket < 0:
        return None
    try:
        edits, _end = json.JSONDecoder().raw_decode(text[bracket:])
    except json.JSONDecodeError:
        try:
            edits, _end = json.JSONDecoder().raw_decode(_close_truncated_json(text[bracket:]))
        except json.JSONDecodeError:
            return None
    return edits if isinstance(edits, list) else None


def parse_or_repair_patch(text: str) -> Dict[str, Any]:
    """Parse patch JSON; if truncated, recover edits[] then close braces."""
    edits = _extract_edits_array(text)
    try:
        obj = extract_json_obj(text)
        if obj.get("edits"):
            return obj
        if edits:
            obj["edits"] = edits
            obj["_repaired"] = True
            return obj
        return obj
    except ValueError:
        pass
    if edits:
        return {
            "edits": edits,
            "claim": "repaired truncated JSON",
            "expect_delta": 0.05,
            "reason": "truncated",
            "_repaired": True,
        }
    try:
        return extract_json_obj(_close_truncated_json(text))
    except ValueError as e:
        raise ValueError(f"unrepairable truncated JSON ({e})") from e


def _image_to_data_url(path: str, *, max_side: int = 384, quality: int = 80) -> Optional[str]:
    """Load POV frame → compressed JPEG data URL for LLMAPI multimodal."""
    try:
        from PIL import Image
    except Exception:
        return None
    if not path or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_side) / float(max(w, h, 1)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def collect_eval_frames(
    img_root: str,
    *,
    max_frames: int = 4,
    prefer_failed: bool = True,
) -> List[str]:
    """Pick a few POV frames from the newest MineStudio image dump dirs.

    Samples early / mid / late steps so the VLM proposer can see orientation
    and whether the agent faces blocks vs sky/walls.
    """
    root = (img_root or "").strip()
    if not root or not os.path.isdir(root):
        return []
    dirs = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("task_"):
            dirs.append(p)
    if not dirs:
        # also allow flat png dumps
        pngs = [
            os.path.join(root, n)
            for n in os.listdir(root)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        pngs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return pngs[: max(1, int(max_frames))]
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    # Prefer most recent episode dumps (failed evals usually last)
    chosen: List[str] = []
    for d in dirs[:3]:
        steps = [
            os.path.join(d, n)
            for n in os.listdir(d)
            if n.startswith("step_") and n.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        steps.sort()
        if not steps:
            continue
        idxs = sorted({0, len(steps) // 2, max(0, len(steps) - 1), max(0, len(steps) // 4)})
        for i in idxs:
            if len(chosen) >= max_frames:
                break
            chosen.append(steps[i])
        if len(chosen) >= max_frames:
            break
    return chosen[: max(1, int(max_frames))]


def build_user_message(
    text: str,
    image_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    paths = [p for p in (image_paths or []) if p]
    if not paths:
        return {"role": "user", "content": text}
    parts: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                text
                + "\n\nAttached POV frames are from the latest sandbox eval "
                "(early/mid/late). Ground your harness edits in what you see. "
                "Do NOT emit game keypresses."
            ),
        }
    ]
    n_img = 0
    for p in paths:
        url = _image_to_data_url(p)
        if not url:
            continue
        parts.append({"type": "image_url", "image_url": {"url": url}})
        n_img += 1
    if n_img == 0:
        return {"role": "user", "content": text}
    return {"role": "user", "content": parts}


def llm_api_chat(
    *,
    messages: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash",
    api_key: str = "",
    request_url: str = "https://api.example.com/v1/chat/completions",
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    reply = llm_api_chat_ex(
        messages=messages,
        model=model,
        api_key=api_key,
        request_url=request_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return str(reply["text"])


def llm_api_chat_ex(
    *,
    messages: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash",
    api_key: str = "",
    request_url: str = "https://api.example.com/v1/chat/completions",
    max_tokens: int = 4096,
    temperature: float = 0.2,
    max_http_retries: int = 6,
) -> Dict[str, Any]:
    from curriculum.gemini_vla import _resolve_app_id

    key = _resolve_app_id(api_key)
    body = {
        "model": model,
        "stream": False,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": messages,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    retries = max(1, int(max_http_retries))
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(
            request_url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            last_err = e
            # Rate-limit / transient gateway — backoff and retry
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                sleep_s = min(90.0, (2 ** attempt) + random.random())
                print(
                    f"[llm_api] HTTP {e.code} retry {attempt + 1}/{retries} "
                    f"sleep={sleep_s:.1f}s body={err_body[:120]!r}",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
            raise RuntimeError(
                f"LLM API HTTP {e.code}: {err_body or e.reason}"
            ) from e
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
            if attempt + 1 < retries:
                sleep_s = min(60.0, (2 ** attempt) + random.random())
                print(
                    f"[llm_api] network error retry {attempt + 1}/{retries} "
                    f"sleep={sleep_s:.1f}s err={e}",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"LLMAPI network failed: {e}") from e

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM API empty choices: {json.dumps(payload)[:400]}")
        choice = choices[0] or {}
        content = (choice.get("message") or {}).get("content") or ""
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text") or ""))
                elif isinstance(p, str):
                    parts.append(p)
            content = "\n".join(parts)
        text = str(content).strip()
        finish = str(choice.get("finish_reason") or "")
        usage = payload.get("usage") or {}
        if not text:
            raise RuntimeError(f"LLM API empty content: {json.dumps(payload)[:400]}")
        return {
            "text": text,
            "finish_reason": finish,
            "usage": usage,
            "truncated": finish.lower() in TRUNC_REASONS,
            "http_attempts": attempt + 1,
        }
    raise RuntimeError(f"LLMAPI failed after retries: {last_err}")


def compact_harness(h: HarnessSpec) -> Dict[str, Any]:
    return {
        "task_prompt": h.task_prompt,
        "decompose_prompt": h.decompose_prompt,
        "recovery_prompt": h.recovery_prompt,
        "probe": h.probe.__dict__,
        "memory": h.memory.__dict__,
        "verify": h.verify.__dict__,
        "recover": h.recover.__dict__,
        "runtime": {
            k: getattr(h.runtime, k)
            for k in (
                "max_steps", "action_chunk_len", "checkpoint_every",
                "instruction", "ticks_per_action",
            )
        },
    }


def failure_summary(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}
    eps = metrics.get("episodes") or []
    n = max(1, len(eps))
    n_fail = sum(1 for e in eps if not e.get("success"))
    n_cascade = sum(1 for e in eps if e.get("cascade_fail"))
    return {
        "score": metrics.get("score"),
        "success_rate": metrics.get("success_rate"),
        "avg_reward": metrics.get("avg_reward"),
        "cascade_rate": metrics.get("cascade_rate"),
        "avg_steps": metrics.get("avg_steps"),
        "eval_seeds": metrics.get("eval_seeds"),
        "baseline_score": metrics.get("baseline_score"),
        "n_episodes": len(eps),
        "n_fail": n_fail,
        "n_cascade": n_cascade,
        "fail_frac": n_fail / n,
        "runtime_stats": metrics.get("runtime_stats"),
        "episodes": [
            {
                "success": e.get("success"),
                "steps": e.get("steps"),
                "reward": e.get("reward"),
                "n_probe": e.get("n_probe"),
                "n_verify_block": e.get("n_verify_block"),
                "n_recover": e.get("n_recover"),
                "cascade_fail": e.get("cascade_fail"),
                "meta": e.get("meta"),
            }
            for e in eps[:8]
        ],
    }


def _coerce(old: Any, value: Any) -> Any:
    if isinstance(old, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes")
        return bool(value)
    if isinstance(old, int) and not isinstance(old, bool):
        return int(value)
    if isinstance(old, float):
        return float(value)
    if isinstance(old, list):
        if not isinstance(value, list):
            return old
        return [str(x) for x in value]
    if isinstance(old, str):
        return str(value)
    return value


def validate_path(path: Sequence[str]) -> Tuple[str, ...]:
    p = tuple(str(x) for x in path)
    if not p:
        raise ValueError("empty edit path")
    if len(p) == 1:
        if p[0] not in ALLOWED_TOP:
            raise ValueError(f"frozen/unknown top path: {p}")
        return p
    head, tail = p[0], p[1]
    if head == "runtime" and tail in FROZEN_RUNTIME:
        raise ValueError(f"frozen runtime field: {tail}")
    allowed = ALLOWED_NESTED.get(head)
    if not allowed or tail not in allowed:
        raise ValueError(f"not editable: {p}")
    if len(p) != 2:
        raise ValueError(f"path too deep: {p}")
    return p


def apply_patch(harness: HarnessSpec, patch: Dict[str, Any]) -> Tuple[HarnessSpec, List[Dict[str, Any]]]:
    h = harness.clone(bump_version=True)
    edits = patch.get("edits")
    if not isinstance(edits, list) or not edits:
        if patch.get("path") and "value" in patch:
            edits = [{"path": patch["path"], "value": patch["value"]}]
        else:
            raise ValueError("patch missing edits[]")
    applied: List[Dict[str, Any]] = []
    for ed in edits[:8]:
        if not isinstance(ed, dict):
            continue
        path = validate_path(list(ed.get("path") or []))
        if len(path) == 1:
            old = getattr(h, path[0])
            new = _coerce(old, ed.get("value"))
            if path[0] in ALLOWED_TOP and isinstance(new, str) and len(new) > 2000:
                new = new[:2000]
            setattr(h, path[0], new)
        else:
            obj = getattr(h, path[0])
            old = getattr(obj, path[1])
            new = _coerce(old, ed.get("value"))
            if path[0] == "probe" and path[1] == "action_pool":
                new = [a for a in new if a in ACTION_NAMES]
                if not new:
                    raise ValueError("empty/invalid action_pool")
            if path[0] == "probe" and path[1] == "selection_strategy":
                new = str(new).lower()
                if new not in ("random", "coverage", "ucb"):
                    raise ValueError("selection_strategy must be random|coverage|ucb")
            if path[0] == "probe" and path[1] in (
                "ucb_c", "transfer_weight", "progress_reward_weight",
                "env_reward_weight", "novelty_reward_weight", "probe_cost",
                "failure_penalty", "downstream_success_reward", "downstream_discount",
            ):
                new = float(new)
                lo, hi = (0.0, 1.0) if path[1] == "downstream_discount" else (0.0, 10.0)
                new = min(max(new, lo), hi)
            if path[0] == "probe" and path[1] == "every_n_steps":
                new = int(min(max(int(new), 1), 32))
            if path[0] == "probe" and path[1] == "budget_per_episode":
                new = int(min(max(int(new), 0), 24))
            if path[0] == "probe" and path[1] == "stall_steps":
                new = int(min(max(int(new), 2), 40))
            if path[0] == "probe" and path[1] == "warmup_probes":
                new = int(min(max(int(new), 0), 8))
            if path[0] == "recover" and path[1] == "max_retries":
                new = int(min(max(int(new), 0), 8))
            if path[0] == "recover" and path[1] == "stall_steps":
                new = int(min(max(int(new), 4), 48))
            if path[0] == "memory" and path[1] == "retrieve_topk":
                new = int(min(max(int(new), 0), 8))
            if path[0] == "runtime" and path[1] == "max_steps":
                new = int(min(max(int(new), 8), 100))
            if path[0] == "runtime" and path[1] == "checkpoint_every":
                new = int(min(max(int(new), 1), 40))
            setattr(obj, path[1], new)
        applied.append({"path": list(path), "before": old, "after": new})
    if not applied:
        raise ValueError("no valid edits applied")
    return h, applied


def _reverted_strategies(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in history or []:
        if rec.get("accepted"):
            continue
        man = rec.get("manifest") or {}
        diag = str(man.get("diagnosis") or "").strip()
        prop = str(man.get("proposal") or "").strip()
        if not diag and not prop:
            continue
        out.append(
            {
                "round": rec.get("round"),
                "diagnosis": diag[:240],
                "proposal": prop[:240],
                "claim": str(man.get("claim") or "")[:120],
            }
        )
    return out[-6:]


def _strategy_hints(
    metrics: Optional[Dict[str, Any]],
    *,
    actionable: bool = False,
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Push proposer toward diagnosis→proposal via prompts, not knob tweaks."""
    m = metrics or {}
    sr = float(m.get("success_rate") or 0.0)
    lines = [
        "Meaningful patch checklist:",
        "  1) diagnosis: concrete POV/state failure (facing sky, wrong block, stuck in wall, ...)",
        "  2) why: why Qwen's current behavior fails",
        "  3) proposal: what Qwen should do (look down, approach target, back off then attack, ...)",
        "  4) edits MUST include >=1 of: task_prompt, recovery_prompt, runtime.instruction",
        "  Optional secondary knobs only AFTER a prompt edit.",
        "Forbidden: ONLY max_retries / budget / action_pool / stall_steps with no prompt change.",
    ]
    if actionable:
        lines.extend(
            [
                "Actionable constraints:",
                "  - Screenshots are PARTIAL; do not conclude the task is impossible.",
                "  - NEVER tell Qwen to stop, give up, or report inability.",
                "  - Proposal must be executable (look/scan/aim/center/approach/attack/enter portal).",
                "  - Do not emit keyPress / w/a/s/d; encode behavior in instruction text.",
            ]
        )
        reverted = _reverted_strategies(history)
        if reverted:
            lines.append("REVERTED strategies (do not repeat diagnosis or proposal):")
            lines.append(json.dumps(reverted, ensure_ascii=False)[:1200])
    if sr < 0.05:
        lines.insert(
            0,
            "CRITICAL: seed success≈0 — diagnose the visual failure mode, then rewrite "
            "instruction/recovery_prompt so Qwen changes behavior.",
        )
    elif sr < 0.5:
        lines.insert(
            0,
            "Partial success — tighten proposal for the remaining failure mode via prompts.",
        )
    return "\n".join(lines)


def _user_prompt(
    harness: HarnessSpec,
    metrics: Optional[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]],
    success_memory: Optional[List[Dict[str, Any]]] = None,
    research_evidence: Optional[Dict[str, Any]] = None,
    actionable: bool = False,
) -> str:
    hist_slim = []
    for rec in (history or [])[-6:]:
        hist_slim.append(
            {
                "round": rec.get("round"),
                "accepted": rec.get("accepted"),
                "score": rec.get("score"),
                "mean_delta": (rec.get("paired") or {}).get("mean_delta"),
                "success_rate": rec.get("success_rate"),
                "manifest": (rec.get("manifest") or {}).get("claim")
                or (rec.get("manifest") or {}).get("op"),
                "diagnosis": (rec.get("manifest") or {}).get("diagnosis"),
                "proposal": (rec.get("manifest") or {}).get("proposal"),
            }
        )
    from curriculum.harness_success_memory import format_success_memory_for_prompt
    from curriculum.harness_research_evidence import format_research_evidence_for_prompt

    mem_txt = format_success_memory_for_prompt(list(success_memory or []))
    evid_txt = format_research_evidence_for_prompt(research_evidence)
    extra = ""
    if actionable:
        extra = (
            "OpenHA init is assumed sufficient. If the POV looks like Overworld while "
            "the goal is Nether, propose look-around / find and enter the portal / "
            "reorient then mine — not abort.\n\n"
        )
    return (
        "Propose ONE meaningful harness patch for a frozen Qwen-VLA in Minecraft.\n"
        "Do not change the VLA weights. Do not emit actions.\n"
        f"{extra}"
        "Priority: research evidence → diagnosis → why → proposal → prompt edits.\n\n"
        f"RESEARCH EVIDENCE (clusters + attribution probes; ground your edit here):\n{evid_txt}\n\n"
        f"Current harness:\n{json.dumps(compact_harness(harness), ensure_ascii=False, default=str)[:3500]}\n\n"
        f"Sandbox eval / failure summary:\n{json.dumps(failure_summary(metrics), ensure_ascii=False, default=str)[:2500]}\n\n"
        f"Prior search history:\n{json.dumps(hist_slim, ensure_ascii=False)[:1500]}\n\n"
        f"Cross-task SUCCESS MEMORY (transferable skills; instantiate {{target}} "
        f"for THIS task; do not copy another task's block name or fixed mouseMove):\n"
        f"{mem_txt}\n\n"
        f"{_strategy_hints(metrics, actionable=actionable, history=history)}\n\n"
        "Editable paths (prompt paths required):\n"
        "- REQUIRED: task_prompt, decompose_prompt, recovery_prompt, runtime.instruction\n"
        "- optional secondary: probe/memory/verify/recover knobs, runtime.max_steps\n\n"
        "Return JSON with these fields. Example:\n"
        '{"diagnosis":"Agent faces open sky, target block not in view",'
        '"why":"Qwen keeps moving forward without looking down to the ore",'
        '"proposal":"Look down, center the target block, then attack",'
        '"edits":['
        '{"path":["runtime","instruction"],"value":"Look down to find the target block, center it, then attack."},'
        '{"path":["recovery_prompt"],"value":"If stuck looking at sky/wall: output the recovery plan as a single <actions> block.\\n<actions> mouseMove(0, 80) ; mouseMove(0, 40) ; keyPress(w) ; mouseClick(left) </actions>"}'
        '],"claim":"look-down then mine","expect_delta":0.05}\n'
    )


def propose_patch(
    harness: HarnessSpec,
    metrics: Optional[Dict[str, Any]],
    *,
    model: str = "gemini-2.5-flash",
    history: Optional[List[Dict[str, Any]]] = None,
    success_memory: Optional[List[Dict[str, Any]]] = None,
    research_evidence: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    temperature: float = 0.0,
    image_paths: Optional[Sequence[str]] = None,
    actionable_patches: bool = False,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Call LLMAPI (optionally multimodal); return (patch, raw, meta)."""
    user = _user_prompt(
        harness,
        metrics,
        history,
        success_memory=success_memory,
        research_evidence=research_evidence,
        actionable=bool(actionable_patches),
    )
    paths = [str(p) for p in (image_paths or []) if p]
    user_msg = build_user_message(user, paths)
    system = SYSTEM + (SYSTEM_ACTIONABLE if actionable_patches else "")
    messages = [
        {"role": "system", "content": system},
        user_msg,
    ]
    last_err: Optional[Exception] = None
    raw = ""
    meta: Dict[str, Any] = {}
    base_temp = float(temperature)
    for attempt in range(max(1, int(max_retries))):
        # Stay greedy on first try; only bump if parse fails (exploration for repair).
        if attempt == 0:
            temp = base_temp
        else:
            temp = min(0.6, max(base_temp, 0.0) + 0.15 * attempt)
        reply = llm_api_chat_ex(
            messages=messages,
            model=model,
            max_tokens=4096,
            temperature=temp,
        )
        raw = str(reply["text"])
        meta = {
            "finish_reason": reply.get("finish_reason"),
            "truncated": bool(reply.get("truncated")),
            "usage": reply.get("usage") or {},
            "attempt": attempt + 1,
            "n_images": len(paths),
            "vision": bool(paths),
            "image_paths": paths[:8],
            "has_research_evidence": bool(research_evidence),
        }
        try:
            patch = parse_or_repair_patch(raw)
            patch = normalize_meaningful_fields(patch)
            if not patch.get("edits") and not (patch.get("component") and patch.get("op")):
                raise ValueError("patch missing edits[]")
            # Bank-style component/op picks skip prompt validation.
            if patch.get("edits"):
                validate_meaningful_patch(patch)
                if actionable_patches:
                    validate_actionable_patch(patch)
        except Exception as e:
            last_err = e
            repair = (
                "Invalid/incomplete patch. Reply with COMPLETE JSON including "
                "diagnosis, why, proposal, and edits[] that MUST change "
                "task_prompt OR recovery_prompt OR runtime.instruction. "
                "Ground diagnosis in RESEARCH EVIDENCE when present. "
            )
            if actionable_patches:
                repair += (
                    "Do NOT tell Qwen to stop or report inability. "
                    "Proposal must be executable look/approach/aim/attack behavior. "
                )
            messages = [
                {"role": "system", "content": system},
                user_msg,
                {"role": "assistant", "content": raw[:800]},
                {
                    "role": "user",
                    "content": repair + f"Error: {e}",
                },
            ]
            continue
        if isinstance(patch.get("claim"), str):
            patch["claim"] = patch["claim"][:120]
        if isinstance(patch.get("reason"), str):
            patch["reason"] = patch["reason"][:80]
        meta["repaired"] = bool(patch.pop("_repaired", False))
        meta["meaningful"] = True
        meta["has_prompt_edit"] = patch_has_prompt_edit(patch)
        meta["actionable_patches"] = bool(actionable_patches)
        return patch, raw, meta
    raise ValueError(f"proposer failed after retries ({last_err}): {raw[:400]!r}")
