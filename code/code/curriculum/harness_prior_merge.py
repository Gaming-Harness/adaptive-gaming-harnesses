"""Accumulate effective harness edits into a stable prior (anti-pollution).

Policy (meaningful merge):
  - Freeze prior_base; do not replace wholesale with latest H*.
  - Transfer promoted *skills* (failure mode → template), not task scripts.
  - Do NOT copy task-locked recovery_prompt / instruction / mouseMove dumps.
  - Do NOT treat max_retries / action_pool / stall knobs as the main transfer.
  - Never inherit task-specific runtime (task_config / instruction paths / img).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from curriculum.harness_schema import HarnessSpec


# Runtime keys that must stay task-local / env-local.
_RUNTIME_BLOCK = {
    "task_config",
    "img_save_dir",
    "instruction",
    "backend",
    "vla_mode",
    "vla_model_path",
    "vla_device",
    "vla_dtype",
    "gemini_api_key",
}


def _uniq(xs: List[str]) -> List[str]:
    out: List[str] = []
    for x in xs:
        s = str(x)
        if s not in out:
            out.append(s)
    return out


def merge_harness_prior(
    base: HarnessSpec,
    incoming: HarnessSpec,
) -> Tuple[HarnessSpec, Dict[str, Any]]:
    """Return (merged, report). ``base`` is preserved as foundation.

    Meaningful transfer = proposals/prompts. Knob merges are secondary and
    limited to enabling capabilities (OR), not retry/pool inflation.
    """
    out = HarnessSpec.from_dict(base.to_dict())
    out.name = "prior_merged"
    out.version = int(getattr(base, "version", 0) or 0) + 1
    report: Dict[str, Any] = {"applied": [], "kept_base": [], "skipped_knobs": []}

    # ---- capability enables (safe OR) ----
    bp, ip = out.probe, incoming.probe
    if ip.enabled and not bp.enabled:
        bp.enabled = True
        report["applied"].append("probe.enabled=True")
    if bp.probe_periodic and not ip.probe_periodic:
        bp.probe_periodic = False
        report["applied"].append("probe.probe_periodic=False")
    for flag in ("probe_on_fail", "probe_on_stall", "prefer_uncertain", "write_on_pass"):
        if bool(getattr(ip, flag, False)) and not bool(getattr(bp, flag, False)):
            setattr(bp, flag, True)
            report["applied"].append(f"probe.{flag}=True")
    # Explicitly do NOT merge budget / stall / action_pool (not human-meaningful).
    report["skipped_knobs"].extend(
        ["probe.budget", "probe.stall_steps", "probe.action_pool", "probe.warmup_probes"]
    )

    br, ir = out.recover, incoming.recover
    if ir.enabled and not br.enabled:
        br.enabled = True
        report["applied"].append("recover.enabled=True")
    for flag in ("use_memory_checkpoint", "replan_with_prompt", "recover_on_stall"):
        if bool(getattr(ir, flag, False)) and not bool(getattr(br, flag, False)):
            setattr(br, flag, True)
            report["applied"].append(f"recover.{flag}=True")
    report["skipped_knobs"].extend(["recover.max_retries", "recover.stall_steps"])

    bm, im = out.memory, incoming.memory
    if im.enabled and not bm.enabled:
        bm.enabled = True
        report["applied"].append("memory.enabled=True")
    for flag in (
        "inject_into_prompt",
        "write_success",
        "write_failure",
        "write_probe",
        "decay_unused",
    ):
        if bool(getattr(im, flag, False)) and not bool(getattr(bm, flag, False)):
            setattr(bm, flag, True)
            report["applied"].append(f"memory.{flag}=True")

    report["kept_base"].append("verify.*")

    # ---- Meaningful: promoted skills only (not task-locked recovery/proposal) ----
    from curriculum.harness_skill_bank import (
        abstract_text,
        extract_skill,
        parse_task_ref,
    )

    in_meta = dict(incoming.meta or {})
    task_file = str(in_meta.get("merged_from_task") or incoming.name or "")
    ref = parse_task_ref(task_file)
    diagnosis = abstract_text(
        str(in_meta.get("last_diagnosis") or "").strip(), target=ref["target"]
    )
    why = abstract_text(
        str(in_meta.get("last_why") or "").strip(), target=ref["target"]
    )
    proposal = abstract_text(
        str(in_meta.get("last_proposal") or "").strip(), target=ref["target"]
    )

    incoming_skills = list(in_meta.get("prior_skills") or [])
    extracted = extract_skill(
        task_file=task_file,
        summary={
            "accepted": 1,
            "holdout_success_rate": in_meta.get("holdout_success_rate") or 0,
            "holdout_win": in_meta.get("holdout_win"),
        },
        harness=incoming,
        diagnosis=diagnosis,
        why=why,
        proposal=proposal,
    )
    if extracted:
        incoming_skills.append(extracted)

    skills = list((out.meta or {}).get("prior_skills") or [])
    by_id = {str(s.get("skill_id")): dict(s) for s in skills if s.get("skill_id")}
    for s in incoming_skills:
        sid = str(s.get("skill_id") or "")
        if not sid:
            continue
        srcs = list((by_id.get(sid) or {}).get("source_tasks") or [])
        for t in s.get("source_tasks") or [task_file]:
            bn = os.path.basename(str(t))
            if bn and bn not in srcs:
                srcs.append(bn)
        promoted = bool(s.get("promoted")) or bool(s.get("holdout_win")) or len(srcs) >= 2
        if not promoted:
            report["skipped_knobs"].append(f"skill:{sid}:candidate")
            continue
        by_id[sid] = {
            "skill_id": sid,
            "triggers": s.get("triggers"),
            "template": s.get("template"),
            "semantic_actions": s.get("semantic_actions"),
            "promoted": True,
            "source_tasks": srcs,
            "n_tasks": len(srcs),
        }
        report["applied"].append(f"meta.prior_skills+={sid}")
    skills = list(by_id.values())[-12:]

    lessons = list((out.meta or {}).get("prior_lessons") or [])
    if extracted and (extracted.get("promoted") or extracted.get("holdout_win")):
        lesson = {
            "skill_id": extracted.get("skill_id"),
            "diagnosis": (extracted.get("diagnosis") or diagnosis)[:200],
            "proposal": (extracted.get("proposal") or extracted.get("template") or "")[:240],
            "from": os.path.basename(task_file),
        }
        if not any(
            (x.get("skill_id") or x.get("proposal") or "")[:60]
            == (lesson.get("skill_id") or lesson.get("proposal") or "")[:60]
            for x in lessons
        ):
            lessons.append(lesson)
            lessons = lessons[-12:]
            report["applied"].append("meta.prior_lessons+=1")

    tips = [
        str(s.get("template") or "")
        for s in skills
        if s.get("template")
    ][-8:]

    # Never copy incoming task-locked recovery onto H0.
    report["kept_base"].append("recovery_prompt")

    report["kept_base"].extend(["task_prompt", "decompose_prompt"])

    sources = list((out.meta or {}).get("prior_sources") or [])
    src_name = str(in_meta.get("warm_start_from") or incoming.name or "")
    fp = incoming.fingerprint()
    sources.append({"name": incoming.name, "fp": fp, "src": src_name})
    sources = sources[-16:]
    out.meta = {
        **dict(out.meta or {}),
        "prior_merge": True,
        "prior_tips": tips,
        "prior_lessons": lessons,
        "prior_skills": skills,
        "prior_sources": sources,
        "last_merged_fp": fp,
        "last_diagnosis": diagnosis,
        "last_why": why,
        "last_proposal": proposal or ((extracted or {}).get("template") or ""),
    }

    report["kept_base"].append("runtime.*")
    _ = _RUNTIME_BLOCK  # documented blocklist; runtime kept as base
    return out, report


def promote_merge(
    *,
    prior_path: str,
    base_path: str,
    best_path: str,
    task_file: str,
    holdout_success_rate: float = 0.0,
    holdout_win: bool = False,
) -> Tuple[HarnessSpec, Dict[str, Any]]:
    """Merge effective ``best`` into existing prior; freeze base on first call.

    - First effective with no base: base := current prior if exists else best
      (foundation), then prior := merge(base, best)  [best may equal base]
    - Later: base unchanged; prior := merge(prior_or_base, best)
    """
    import os

    best = HarnessSpec.load(best_path)
    if not os.path.isfile(base_path):
        if os.path.isfile(prior_path):
            foundation = HarnessSpec.load(prior_path)
        else:
            foundation = HarnessSpec.load(best_path)
        foundation.name = "prior_base"
        foundation.meta = {
            **dict(foundation.meta or {}),
            "prior_base": True,
            "base_from_task": task_file,
        }
        foundation.save(base_path)
        base = foundation
    else:
        base = HarnessSpec.load(base_path)

    if os.path.isfile(prior_path):
        cur = HarnessSpec.load(prior_path)
    else:
        cur = base

    # Annotate incoming with task id for lesson bank.
    best.meta = {
        **dict(best.meta or {}),
        "merged_from_task": task_file,
        "holdout_success_rate": float(holdout_success_rate or 0.0),
        "holdout_win": bool(holdout_win) or float(holdout_success_rate or 0.0) >= 0.5,
    }
    merged, report = merge_harness_prior(cur, best)
    merged.meta = {
        **dict(merged.meta or {}),
        "prior_base_fp": base.fingerprint(),
        "merged_from_task": task_file,
        "merged_from_best_fp": best.fingerprint(),
    }
    merged.save(prior_path)
    report["base_path"] = base_path
    report["prior_path"] = prior_path
    report["base_fp"] = base.fingerprint()
    report["prior_fp"] = merged.fingerprint()
    report["best_fp"] = best.fingerprint()
    return merged, report
