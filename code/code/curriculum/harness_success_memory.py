"""Cross-task success memory for Auto-Harness (does not change H0).

Successful / effective harnesses are appended here. The next task still starts
from the original prior base; the proposer only *reads* this memory.

Lessons are transferable skills (failure mode → template), not task scripts.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from curriculum.harness_schema import HarnessSpec


def success_memory_path(out_root: str) -> str:
    return os.path.join(out_root, "success_memory.jsonl")


def load_success_memory(path: str, *, limit: int = 24) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-max(1, int(limit)) :]


def _lesson_from_logdir(logdir: str) -> Dict[str, str]:
    """Best-effort: pull diagnosis/proposal from last accepted llm_patch / history."""
    out: Dict[str, str] = {}
    hist_p = os.path.join(logdir, "history.json")
    if os.path.isfile(hist_p):
        try:
            hist = json.load(open(hist_p))
            for rec in reversed(hist or []):
                if not rec.get("accepted"):
                    continue
                man = rec.get("manifest") or {}
                for k in ("diagnosis", "why", "proposal", "claim"):
                    if man.get(k) and not out.get(k):
                        out[k] = str(man[k])[:320]
                if out.get("diagnosis") or out.get("proposal"):
                    break
        except Exception:
            pass
    # fallback: newest llm_patch_r*.json
    if not out.get("diagnosis"):
        try:
            patches = sorted(
                [
                    os.path.join(logdir, fn)
                    for fn in os.listdir(logdir)
                    if fn.startswith("llm_patch_r") and fn.endswith(".json")
                ]
            )
            for p in reversed(patches):
                try:
                    patch = json.load(open(p))
                except Exception:
                    continue
                for k in ("diagnosis", "why", "proposal", "claim"):
                    if patch.get(k) and not out.get(k):
                        out[k] = str(patch[k])[:320]
                if out.get("diagnosis") or out.get("proposal"):
                    break
        except Exception:
            pass
    return out


def compact_success_entry(
    *,
    task_file: str,
    logdir: str,
    summary: Dict[str, Any],
    harness: HarnessSpec,
) -> Dict[str, Any]:
    """Store transferable lessons, not full task runtime."""
    from curriculum.harness_skill_bank import abstract_text, extract_skill, parse_task_ref

    h = harness
    meta = dict(h.meta or {})
    from_log = _lesson_from_logdir(logdir)
    diagnosis = str(
        meta.get("last_diagnosis") or from_log.get("diagnosis") or ""
    )[:240]
    why = str(meta.get("last_why") or from_log.get("why") or "")[:240]
    proposal = str(
        meta.get("last_proposal") or from_log.get("proposal") or from_log.get("claim") or ""
    )[:320]
    ref = parse_task_ref(task_file)
    skill = extract_skill(
        task_file=task_file,
        logdir=logdir,
        summary=summary,
        harness=h,
        diagnosis=diagnosis,
        why=why,
        proposal=proposal,
    )
    abs_diag = abstract_text(diagnosis, target=ref["target"])[:240]
    abs_why = abstract_text(why, target=ref["target"])[:240]
    abs_prop = abstract_text(proposal, target=ref["target"])[:320]
    return {
        "task": task_file,
        "logdir": logdir,
        "timestamp": time.time(),
        "accepted": int(summary.get("accepted") or 0),
        "seed_score": summary.get("seed_score"),
        "best_score": summary.get("best_score"),
        "holdout_success_rate": summary.get("holdout_success_rate"),
        "harness_fp": h.fingerprint(),
        "skill": skill,
        "lesson": {
            "diagnosis": abs_diag,
            "why": abs_why,
            "proposal": abs_prop,
            "skill_id": (skill or {}).get("skill_id"),
            "template": (skill or {}).get("template"),
            "triggers": (skill or {}).get("triggers"),
            "recovery_prompt_snip": abstract_text(
                str(h.recovery_prompt or ""), target=ref["target"]
            )[:220],
            "instruction_snip": abstract_text(
                str(getattr(h.runtime, "instruction", "") or ""),
                target=ref["target"],
            )[:180],
            "probe_on_stall": bool(h.probe.probe_on_stall),
            "probe_on_fail": bool(h.probe.probe_on_fail),
            "action_pool": list(h.probe.action_pool or [])[:12],
            "max_retries": int(h.recover.max_retries),
        },
    }


def append_success_memory(path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def format_success_memory_for_prompt(
    rows: List[Dict[str, Any]], *, max_chars: int = 2000
) -> str:
    if not rows:
        return "(empty)"
    from curriculum.harness_skill_bank import abstract_text, parse_task_ref

    slim = []
    for r in rows[-12:]:
        lesson = r.get("lesson") or {}
        skill = r.get("skill") or {}
        ref = parse_task_ref(str(r.get("task") or ""))
        tmpl = skill.get("template") or lesson.get("template")
        slim.append({
            "skill_id": skill.get("skill_id") or lesson.get("skill_id"),
            "triggers": skill.get("triggers") or lesson.get("triggers"),
            "template": tmpl,
            "promoted": bool(skill.get("promoted")),
            "hold": r.get("holdout_success_rate"),
            "diagnosis": abstract_text(
                str(lesson.get("diagnosis") or ""), target=ref["target"]
            ) or None,
            "proposal": abstract_text(
                str(lesson.get("proposal") or lesson.get("tip") or ""),
                target=ref["target"],
            ) or None,
        })
    text = json.dumps(slim, ensure_ascii=False, default=str)
    return text[:max_chars]


def ensure_original_prior_base(base_path: str) -> HarnessSpec:
    """Write a clean fail/stall-only strong seed as the immutable H0 base."""
    from curriculum.auto_harness_search import strong_seed

    if os.path.isfile(base_path):
        return HarnessSpec.load(base_path)
    h = strong_seed()
    h.name = "prior_base_original"
    h.probe.probe_periodic = False
    h.probe.probe_on_fail = True
    h.probe.probe_on_stall = True
    h.probe.stall_steps = 8
    h.probe.warmup_probes = 0
    h.probe.budget_per_episode = 8
    h.probe.action_pool = [
        "forward", "attack", "forward_attack", "turn_left", "look_up", "look_down",
    ]
    h.verify.enabled = False
    h.verify.prefer_memory_action = False
    h.recover.enabled = True
    h.recover.recover_on_stall = True
    h.recover.stall_steps = 12
    h.recover.max_retries = 2
    h.meta = {"prior_base": True, "immutable_h0": True}
    h.save(base_path)
    return h
