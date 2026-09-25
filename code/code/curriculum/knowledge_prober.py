"""Knowledge-level probing: choose which transferable skill to inject.

The VLA remains frozen. Only contextual bandit statistics over skill IDs evolve.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def knowledge_context(harness: Any, tags: Sequence[str]) -> str:
    meta = dict(getattr(harness, "meta", None) or {})
    task_type = str(meta.get("current_task_type") or "unknown")
    stable = sorted({str(t) for t in tags if str(t)})
    return f"{task_type}:{'+'.join(stable) if stable else 'recover'}"


def select_knowledge(
    harness: Any, *, tags: Sequence[str], bandit: Any, rng: Any, target: str = ""
) -> Optional[Dict[str, Any]]:
    """Select one knowledge arm and bind its abstract template to this task."""
    meta = dict(getattr(harness, "meta", None) or {})
    candidates = list(meta.get("knowledge_candidates") or meta.get("prior_skills") or [])
    candidates = [c for c in candidates if c.get("skill_id") and c.get("template")]
    if not candidates:
        return None
    # Task compatibility is filtered when candidates are overlaid. Triggers
    # belong in the context, not as a hard filter: otherwise a context often
    # has one arm and the prober cannot discover an unexpected transfer.
    pool = candidates
    context = knowledge_context(harness, tags)
    strategy = str(
        getattr(harness.knowledge_probe, "selection_strategy", "ucb")
    ).lower()
    arms = [str(c["skill_id"]) for c in pool]
    arm = bandit.select(context, arms, rng) if strategy == "ucb" else arms[0]
    picked = next(c for c in pool if str(c["skill_id"]) == arm)
    tgt = target or str(meta.get("current_target") or "target")
    return {
        "skill_id": arm,
        "context": context,
        "text": str(picked["template"]).replace("{target}", tgt),
        "candidate_ids": arms,
        "promoted": bool(picked.get("promoted")),
        "n_tasks": int(picked.get("n_tasks") or 0),
    }
