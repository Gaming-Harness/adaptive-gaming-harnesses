#!/usr/bin/env python3
"""Harness-VLA agent over existing Qwen ↔ MineStudio sandbox.

RPent pattern, MineStudio stack:
  Planner (LLMAPI Gemini / text LLM)  — slow, tool-level
       └─ tools: view_state / read_memory / qwen_skill / finish
  Qwen-VLA                            — fast skill (action chunks)
  MineStudio sandbox                  — execute + score

Does NOT import RPent/LIBERO. Reuses HarnessRuntime.run_vla_skill.

Usage:
  cd project_root
  python -m curriculum.harness_vla_agent --smoke
  python -m curriculum.harness_vla_agent \\
    --config curriculum/configs/harness_vla_minestudio.yaml
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.harness_runtime import HarnessRuntime, _instruction_from_task, make_env
from curriculum.harness_schema import HarnessSpec, default_seed_harness
from curriculum.auto_harness_search import degraded_seed, strong_seed


TOOLS_SPEC = """
You are a Minecraft Harness planner. You NEVER output low-level keys
(w/a/s/d/attack). You only call tools by returning ONE JSON object:

{"tool": "<name>", "args": {..}, "reason": "<short>"}

Tools:
1) view_state  args: {}
   → brief description of last observation (reward/success/fail flags).
2) read_memory args: {}
   → verified experience snippets from harness memory.
3) qwen_skill  args: {"instruction": "<subgoal>"}
   → run frozen Qwen-VLA for many action chunks (OpenHA-scale budget). Do NOT set a small max_chunks.
4) finish      args: {"success": true|false, "summary": "..."}
   → end the episode.

Prefer short subgoals (look at tree → approach → chop). Reuse memory.
After qwen_skill, check view_state before finishing.
""".strip()


def _llm_api_text(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    request_url: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout_s: float = 90.0,
    min_interval_s: float = 2.0,
    last_ts: List[float],
) -> str:
    if min_interval_s > 0 and last_ts[0] > 0:
        wait = min_interval_s - (time.time() - last_ts[0])
        if wait > 0:
            time.sleep(wait)
    body = {
        "model": model,
        "stream": False,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": messages,
    }
    req = urllib.request.Request(
        request_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    last_ts[0] = time.time()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"planner empty: {json.dumps(payload)[:400]}")
    content = (choices[0].get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "\n".join(
            str(p.get("text") if isinstance(p, dict) else p) for p in content
        )
    return str(content).strip()


def _parse_tool_call(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    # fenced json
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    raw = m.group(1) if m else None
    if raw is None:
        m2 = re.search(r"\{.*\}", text, flags=re.S)
        raw = m2.group(0) if m2 else text
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {
            "tool": "finish",
            "args": {"success": False, "summary": f"bad_planner_json: {e}"},
            "reason": "parse_fail",
            "raw": text[:500],
        }
    tool = str(obj.get("tool") or obj.get("name") or "finish")
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    # allow flat keys
    if not args:
        args = {k: v for k, v in obj.items() if k not in ("tool", "name", "reason")}
    return {
        "tool": tool,
        "args": args,
        "reason": str(obj.get("reason") or ""),
        "raw": text[:500],
    }


@dataclass
class HarnessVLAConfig:
    planner_model: str = "gemini-2.5-flash"
    planner_base_url: str = (
        "https://api.example.com/v1/chat/completions"
    )
    planner_api_key: str = ""
    max_planner_steps: int = 6
    skill_max_chunks: int = 2
    logdir: str = "curriculum/outputs/harness_vla"
    seed: int = 0


@dataclass
class HarnessVLAAgent:
    """Planner (LLM) + HarnessRuntime (Qwen skill) + MineStudio sandbox."""

    harness: HarnessSpec
    cfg: HarnessVLAConfig = field(default_factory=HarnessVLAConfig)
    runtime: Optional[HarnessRuntime] = None

    def __post_init__(self) -> None:
        if self.runtime is None:
            self.runtime = HarnessRuntime(self.harness, seed=int(self.cfg.seed))
        from curriculum.gemini_vla import _resolve_app_id

        self.cfg.planner_api_key = _resolve_app_id(self.cfg.planner_api_key)
        self._last_ts = [0.0]
        self.trace: List[Dict[str, Any]] = []

    def _obs_brief(self, obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(obs, dict):
            return {"empty": True}
        return {
            "reward": obs.get("reward"),
            "success": bool(obs.get("success")),
            "terminated": bool(obs.get("terminated")),
            "last_fail": bool(obs.get("last_fail")),
            "progress": obs.get("progress"),
            "t": obs.get("t"),
            "has_image": obs.get("image") is not None,
        }

    def _memory_brief(self, k: int = 5) -> List[Dict[str, Any]]:
        items = []
        mem = self.runtime.memory
        for it in list(getattr(mem, "items", []) or [])[:k]:
            meta = getattr(it, "meta", {}) or {}
            items.append(
                {
                    "action": meta.get("action"),
                    "status": getattr(it, "status", None),
                    "confidence": float(getattr(it, "confidence", 0.0)),
                    "note": meta.get("note"),
                }
            )
        return items

    def _dispatch(
        self,
        tool: str,
        args: Dict[str, Any],
        *,
        env: Any,
        obs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Return (result, new_obs, done)."""
        tool = str(tool).lower().strip()
        if tool in ("view_state", "view", "state"):
            return {"obs": self._obs_brief(obs)}, obs, False
        if tool in ("read_memory", "memory"):
            return {"memory": self._memory_brief()}, obs, False
        if tool in ("qwen_skill", "vla_skill", "skill"):
            instr = str(args.get("instruction") or args.get("subgoal") or "").strip()
            if not instr:
                instr = str(self.harness.runtime.instruction)
            # Do not let the planner shrink the OpenHA-style budget (was stuck at 2).
            max_chunks = int(self.cfg.skill_max_chunks)
            if args.get("max_chunks") is not None:
                try:
                    asked = int(args.get("max_chunks"))
                    if asked > max_chunks:
                        max_chunks = asked
                except (TypeError, ValueError):
                    pass
            out = self.runtime.run_vla_skill(
                env, instr, max_chunks=max_chunks, obs=obs, use_memory=True
            )
            new_obs = out.get("obs") if isinstance(out.get("obs"), dict) else obs
            slim = {
                "success": out.get("success"),
                "done": out.get("done"),
                "reward": out.get("reward"),
                "chunks_used": out.get("chunks_used"),
                "chunks": out.get("chunks"),
                "instruction": out.get("instruction"),
                "obs": self._obs_brief(new_obs),
            }
            done = bool(out.get("success") or out.get("done"))
            return slim, new_obs, done
        if tool in ("finish", "done", "stop"):
            return {
                "finished": True,
                "success": bool(args.get("success", False)),
                "summary": str(args.get("summary") or ""),
            }, obs, True
        return {"error": f"unknown_tool:{tool}"}, obs, False

    def run(self, *, env: Any = None) -> Dict[str, Any]:
        owns = env is None
        if env is None:
            env = make_env(self.harness, seed=int(self.cfg.seed))
        goal = _instruction_from_task(
            str(self.harness.runtime.task_config or ""),
            str(self.harness.runtime.instruction),
        )
        self.harness.runtime.instruction = goal
        obs = env.reset()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": TOOLS_SPEC},
            {
                "role": "user",
                "content": (
                    f"Task goal: {goal}\n"
                    f"Initial state: {json.dumps(self._obs_brief(obs))}\n"
                    "Call the first tool now."
                ),
            },
        ]
        finished = False
        success = False
        summary = ""
        try:
            for step in range(int(self.cfg.max_planner_steps)):
                text = _llm_api_text(
                    messages=messages,
                    model=self.cfg.planner_model,
                    api_key=self.cfg.planner_api_key,
                    request_url=self.cfg.planner_base_url,
                    last_ts=self._last_ts,
                )
                call = _parse_tool_call(text)
                result, obs, done = self._dispatch(
                    call["tool"], call.get("args") or {}, env=env, obs=obs
                )
                rec = {
                    "step": step,
                    "tool": call["tool"],
                    "args": call.get("args"),
                    "reason": call.get("reason"),
                    "result": result,
                }
                self.trace.append(rec)
                print(
                    f"[harness_vla] step={step} tool={call['tool']} "
                    f"reason={call.get('reason', '')[:80]}",
                    flush=True,
                )
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool result:\n" + json.dumps(result, ensure_ascii=False)[:3500],
                    }
                )
                if call["tool"] in ("finish", "done", "stop"):
                    finished = True
                    success = bool((call.get("args") or {}).get("success", False)) or bool(
                        result.get("success")
                    )
                    summary = str((call.get("args") or {}).get("summary") or "")
                    break
                if done and isinstance(obs, dict) and obs.get("success"):
                    success = True
                    finished = True
                    summary = "env_success"
                    break
        finally:
            if owns and hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass

        out = {
            "success": bool(success),
            "finished": bool(finished),
            "summary": summary,
            "goal": goal,
            "n_planner_steps": len(self.trace),
            "trace": self.trace,
            "memory_items": len(self.runtime.memory),
            "harness_fp": self.harness.fingerprint(),
        }
        return out


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("pyyaml required") from e
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_from_config(cfg: Dict[str, Any]) -> HarnessVLAAgent:
    start = str(cfg.get("start") or "strong")
    if start == "strong":
        h = strong_seed()
    elif start == "degraded":
        h = degraded_seed()
    elif os.path.isfile(start):
        h = HarnessSpec.load(start)
    else:
        h = default_seed_harness()
    rt = h.runtime
    rt.backend = str(cfg.get("backend") or "minestudio")
    rt.vla_mode = str(cfg.get("vla_mode") or "hf")
    if cfg.get("vla_model_path"):
        rt.vla_model_path = str(cfg["vla_model_path"])
    rt.vla_device = str(cfg.get("vla_device") or "cuda")
    rt.vla_dtype = str(cfg.get("vla_dtype") or "bfloat16")
    rt.action_chunk_len = int(cfg.get("action_chunk_len") or 4)
    rt.max_steps = int(cfg.get("max_steps") or 16)
    rt.ticks_per_action = int(cfg.get("ticks_per_action") or 4)
    if cfg.get("task_config"):
        rt.task_config = str(cfg["task_config"])
    if cfg.get("img_save_dir"):
        rt.img_save_dir = str(cfg["img_save_dir"])
    rt.success_reward_thresh = float(cfg.get("success_reward_thresh") or 0.5)
    rt.soft_rollback = bool(cfg.get("soft_rollback", True))
    if cfg.get("instruction"):
        rt.instruction = str(cfg["instruction"])

    # keep probe/verify soft-on so skill path still writes memory
    if "probe_enabled" in cfg:
        h.probe.enabled = bool(cfg["probe_enabled"])
    if "verify_enabled" in cfg:
        h.verify.enabled = bool(cfg["verify_enabled"])
    if "prefer_memory_action" in cfg:
        h.verify.prefer_memory_action = bool(cfg["prefer_memory_action"])
    if cfg.get("max_steps") is not None:
        rt.max_steps = int(cfg["max_steps"])

    agent_cfg = HarnessVLAConfig(
        planner_model=str(cfg.get("planner_model") or "gemini-2.5-flash"),
        planner_base_url=str(
            cfg.get("planner_base_url")
            or "https://api.example.com/v1/chat/completions"
        ),
        planner_api_key=str(cfg.get("planner_api_key") or ""),
        max_planner_steps=int(cfg.get("max_planner_steps") or 6),
        skill_max_chunks=int(cfg.get("skill_max_chunks") or 2),
        logdir=str(cfg.get("logdir") or "curriculum/outputs/harness_vla"),
        seed=int(cfg.get("seed") or 0),
    )
    return HarnessVLAAgent(harness=h, cfg=agent_cfg)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Harness-VLA over Qwen + MineStudio")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--smoke", action="store_true", help="stub env + stub VLA + fake planner finish")
    p.add_argument("--logdir", type=str, default="")
    p.add_argument("--max-planner-steps", type=int, default=0)
    args = p.parse_args(argv)

    if args.smoke:
        h = strong_seed()
        h.runtime.backend = "stub"
        h.runtime.vla_mode = "stub"
        h.runtime.max_steps = 8
        agent = HarnessVLAAgent(
            harness=h,
            cfg=HarnessVLAConfig(max_planner_steps=3, skill_max_chunks=1, seed=0),
        )
        # smoke without LLMAPI: drive tools manually
        env = make_env(h, seed=0)
        obs = env.reset()
        r1, obs, _ = agent._dispatch(
            "qwen_skill",
            {"instruction": "Walk forward toward the goal.", "max_chunks": 1},
            env=env,
            obs=obs,
        )
        agent.trace.append({"step": 0, "tool": "qwen_skill", "result": r1})
        r2, obs, _ = agent._dispatch("view_state", {}, env=env, obs=obs)
        agent.trace.append({"step": 1, "tool": "view_state", "result": r2})
        r3, obs, _ = agent._dispatch(
            "finish", {"success": bool(r1.get("success")), "summary": "smoke"}, env=env, obs=obs
        )
        agent.trace.append({"step": 2, "tool": "finish", "result": r3})
        if hasattr(env, "close"):
            env.close()
        out = {
            "success": bool(r1.get("success") or (isinstance(obs, dict) and obs.get("success"))),
            "mode": "smoke",
            "trace": agent.trace,
        }
        print(json.dumps({k: out[k] for k in out if k != "trace"}, indent=2))
        logdir = args.logdir or "curriculum/outputs/harness_vla_smoke"
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "result.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[harness_vla] wrote {logdir}/result.json", flush=True)
        return 0

    if not args.config:
        p.error("--config required (or --smoke)")
    cfg = _load_yaml(args.config)
    if args.logdir:
        cfg["logdir"] = args.logdir
    if args.max_planner_steps > 0:
        cfg["max_planner_steps"] = args.max_planner_steps
    agent = build_from_config(cfg)
    out = agent.run()
    logdir = str(cfg.get("logdir") or agent.cfg.logdir)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "result.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(
        f"[harness_vla] success={out['success']} steps={out['n_planner_steps']} "
        f"→ {logdir}/result.json",
        flush=True,
    )
    return 0 if out.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
