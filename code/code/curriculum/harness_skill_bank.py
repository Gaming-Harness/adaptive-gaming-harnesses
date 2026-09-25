"""Transferable skill bank for Auto-Harness.

ACCEPT writes a *skill candidate* (failure mode → template), not a task script.
Promote into H0/meta only if holdout wins or the same skill fires on another task.
Runtime binds ``{target}`` for the current task; never copy melon/mouseMove dumps.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

SKILL_CATALOG: Dict[str, Dict[str, Any]] = {
    "reacquire_and_center": {
        "triggers": ["orientation", "target_loss"],
        "task_types": ["mine_block", "kill_entity"],
        "template": (
            "Previous action failed. If the target {target} is not centered in "
            "your view or you are looking away, re-orient to face and center it, "
            "then retry."
        ),
        "semantic_actions": ["look_toward_target", "approach", "attack"],
    },
    "backup_and_turn": {
        "triggers": ["stall_heavy"],
        "task_types": ["mine_block", "craft_item", "kill_entity"],
        "template": (
            "Previous action failed. If you are stuck facing a wall or enclosed, "
            "step back, turn, then retry the current subgoal."
        ),
        "semantic_actions": ["back", "turn", "forward"],
    },
    "approach_and_act": {
        "triggers": ["zero_progress", "near_miss", "execution_gap"],
        "task_types": ["mine_block", "kill_entity", "craft_item"],
        "template": (
            "Previous action failed. Approach the target {target} until it is in "
            "reach, then attack or use."
        ),
        "semantic_actions": ["forward", "attack"],
    },
}

_GENERIC_RECOVERY = (
    "return to the last successful checkpoint",
    "re-evaluate environment and plan new approach",
    "retry the current subgoal with a different action",
)

_MODIFY_PROMPT_RE = re.compile(
    r"modify the\s+`?recovery_prompt`?", re.I
)
_ACTIONS_RE = re.compile(r"<actions>.*?</actions>", re.I | re.S)
_MOUSEMOVE_RE = re.compile(r"mouseMove\s*\([^)]*\)", re.I)


def skill_bank_path(out_root: str) -> str:
    return os.path.join(out_root, "skill_bank.jsonl")


def parse_task_ref(task_file: str) -> Dict[str, str]:
    stem = os.path.basename(str(task_file or "")).replace(".json", "")
    # Evaluation manifests may prefix semantic task names with identifiers such
    # as ``g000_``. Locate the family marker instead of assuming a numeric-only
    # prefix; otherwise held-out tasks silently become ``unknown`` and no
    # promoted skill can be bound to them.
    family = re.search(r"(?:^|_)(mine_block|kill_entity|craft_item)(?:_|$)", stem)
    if family:
        stem = stem[family.start(1) :]
    else:
        stem = re.sub(r"^\d+_", "", stem)
    task_type = "unknown"
    target = stem.replace("_", " ").strip()
    for prefix in ("mine_block", "kill_entity", "craft_item"):
        if stem.startswith(prefix + "_") or stem == prefix:
            task_type = prefix
            target = stem[len(prefix) :].lstrip("_").replace("_", " ").strip()
            break
    return {"task_type": task_type, "target": target or "target", "stem": stem}


def abstract_text(text: str, *, target: str = "", extra_terms: Sequence[str] = ()) -> str:
    """Strip task nouns, action tokens, and proposer-edit meta."""
    t = str(text or "")
    t = _ACTIONS_RE.sub("", t)
    t = _MOUSEMOVE_RE.sub("", t)
    t = _MODIFY_PROMPT_RE.sub("use the skill", t)
    terms: List[str] = []
    for raw in [target, *list(extra_terms)]:
        s = str(raw or "").strip()
        if not s:
            continue
        terms.append(s)
        terms.append(s.replace("_", " "))
        terms.append(s.replace(" ", "_"))
        if s.endswith(" block"):
            terms.append(s[: -len(" block")])
        else:
            terms.append(s + " block")
    terms = sorted({x for x in terms if len(x) >= 3}, key=len, reverse=True)
    for term in terms:
        t = re.sub(re.escape(term), "{target}", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(\{target\}\s*){2,}", "{target} ", t)
    return t


def is_generic_recovery(text: str) -> bool:
    low = str(text or "").lower()
    if not low.strip():
        return True
    if _ACTIONS_RE.search(low) or _MOUSEMOVE_RE.search(low):
        return False
    return any(p in low for p in _GENERIC_RECOVERY) and "center" not in low and "re-orient" not in low


def infer_skill_id(
    tags: Sequence[str],
    *,
    diagnosis: str = "",
    proposal: str = "",
    recovery: str = "",
) -> Optional[str]:
    tagset = {str(t) for t in (tags or [])}
    blob = " ".join([diagnosis, proposal, recovery]).lower()
    if "orientation" in tagset or "target_loss" in tagset:
        return "reacquire_and_center"
    if any(k in blob for k in ("looking away", "look away", "towards the sky", "center", "re-orient", "reorient")):
        return "reacquire_and_center"
    if "stall_heavy" in tagset or any(k in blob for k in ("enclosed", "facing a wall", "stuck inside")):
        return "backup_and_turn"
    if "execution_gap" in tagset or "near_miss" in tagset:
        return "approach_and_act"
    if "zero_progress" in tagset:
        return "approach_and_act"
    if is_generic_recovery(recovery) and not diagnosis.strip():
        return None
    return None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
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
    return rows


def load_skill_bank(path: str) -> List[Dict[str, Any]]:
    return _load_jsonl(path)


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tags_from_logdir(logdir: str) -> List[str]:
    tags: List[str] = []
    for name in ("funnel.json", "summary.json", "attribution_r1.json"):
        p = os.path.join(logdir, name) if logdir else ""
        if not p or not os.path.isfile(p):
            continue
        try:
            obj = json.load(open(p))
        except Exception:
            continue
        for key in ("tags", "this_task_tags", "failure_tags"):
            val = obj.get(key)
            if isinstance(val, list):
                tags.extend(str(x) for x in val)
        evid = obj.get("research_evidence") or obj.get("evidence") or {}
        if isinstance(evid, dict):
            tags.extend(str(x) for x in (evid.get("tags") or []))
    # newest attribution
    if logdir and os.path.isdir(logdir):
        attrs = sorted(
            os.path.join(logdir, fn)
            for fn in os.listdir(logdir)
            if fn.startswith("attribution_r") and fn.endswith(".json")
        )
        for p in reversed(attrs[-3:]):
            try:
                obj = json.load(open(p))
            except Exception:
                continue
            cause = str(obj.get("likely_cause") or "")
            if cause and cause != "unknown":
                tags.append(cause)
            pov = str(obj.get("pov_content") or "").lower()
            if any(k in pov for k in ("sky", "wall", "ceiling")):
                tags.append("orientation")
    out: List[str] = []
    for t in tags:
        if t and t not in out:
            out.append(t)
    return out


def extract_skill(
    *,
    task_file: str,
    logdir: str = "",
    summary: Optional[Dict[str, Any]] = None,
    harness: Any = None,
    tags: Optional[Sequence[str]] = None,
    diagnosis: str = "",
    why: str = "",
    proposal: str = "",
    recovery: str = "",
) -> Optional[Dict[str, Any]]:
    ref = parse_task_ref(task_file)
    summary = summary or {}
    if harness is not None:
        recovery = recovery or str(getattr(harness, "recovery_prompt", "") or "")
        meta = dict(getattr(harness, "meta", None) or {})
        diagnosis = diagnosis or str(meta.get("last_diagnosis") or "")
        why = why or str(meta.get("last_why") or "")
        proposal = proposal or str(meta.get("last_proposal") or "")
    tag_list = list(tags or []) or _tags_from_logdir(logdir)
    skill_id = infer_skill_id(
        tag_list, diagnosis=diagnosis, proposal=proposal, recovery=recovery
    )
    if not skill_id:
        return None
    cat = SKILL_CATALOG[skill_id]
    hold = float(summary.get("holdout_success_rate") or 0.0)
    holdout_win = bool(summary.get("holdout_win")) or hold >= 0.5
    return {
        "skill_id": skill_id,
        "triggers": list(cat["triggers"]),
        "task_types": list(cat["task_types"]),
        "template": cat["template"],
        "semantic_actions": list(cat["semantic_actions"]),
        "diagnosis": abstract_text(diagnosis, target=ref["target"]),
        "why": abstract_text(why, target=ref["target"]),
        "proposal": abstract_text(proposal, target=ref["target"]),
        "from_task": os.path.basename(str(task_file)),
        "task_type": ref["task_type"],
        "holdout_win": holdout_win,
        "holdout_success_rate": hold,
        "accepted": int(summary.get("accepted") or 0),
        "promoted": False,
        "source_tasks": [os.path.basename(str(task_file))],
        "n_tasks": 1,
        "n_accept": max(1, int(summary.get("accepted") or 1)),
        "timestamp": time.time(),
        "raw_recovery_snip": str(recovery or "")[:180],
    }


def upsert_skill(path: str, skill: Dict[str, Any]) -> Dict[str, Any]:
    """Merge into bank. Promote if holdout_win or seen on a second task."""
    rows = load_skill_bank(path)
    sid = str(skill.get("skill_id") or "")
    src = os.path.basename(str((skill.get("source_tasks") or ["?"])[0]))
    found = None
    for row in rows:
        if str(row.get("skill_id") or "") == sid:
            found = row
            break
    if found is None:
        skill = dict(skill)
        skill["promoted"] = bool(skill.get("holdout_win"))
        rows.append(skill)
        found = skill
    else:
        sources = list(found.get("source_tasks") or [])
        if src and src not in sources:
            sources.append(src)
        found["source_tasks"] = sources
        found["n_tasks"] = len(sources)
        found["n_accept"] = int(found.get("n_accept") or 0) + int(skill.get("n_accept") or 1)
        found["holdout_win"] = bool(found.get("holdout_win")) or bool(skill.get("holdout_win"))
        found["timestamp"] = time.time()
        for k in ("diagnosis", "why", "proposal", "template"):
            if skill.get(k) and not found.get(k):
                found[k] = skill[k]
        found["promoted"] = bool(found.get("holdout_win")) or int(found.get("n_tasks") or 1) >= 2
    _write_jsonl(path, rows)
    return dict(found)


def record_skill(
    out_root: str,
    *,
    task_file: str,
    logdir: str = "",
    summary: Optional[Dict[str, Any]] = None,
    harness: Any = None,
    tags: Optional[Sequence[str]] = None,
    diagnosis: str = "",
    why: str = "",
    proposal: str = "",
) -> Optional[Dict[str, Any]]:
    skill = extract_skill(
        task_file=task_file,
        logdir=logdir,
        summary=summary,
        harness=harness,
        tags=tags,
        diagnosis=diagnosis,
        why=why,
        proposal=proposal,
    )
    if not skill:
        return None
    return upsert_skill(skill_bank_path(out_root), skill)


def promoted_skills(
    out_root: str, *, task_type: str = ""
) -> List[Dict[str, Any]]:
    rows = load_skill_bank(skill_bank_path(out_root))
    out = []
    for row in rows:
        if not row.get("promoted"):
            continue
        types = row.get("task_types") or []
        if task_type and types and task_type not in types:
            continue
        out.append(row)
    return out


def hydrate_skill_bank(out_root: str) -> int:
    """Backfill bank from existing success_memory.jsonl (once per missing task)."""
    mem_p = os.path.join(out_root, "success_memory.jsonl")
    bank_p = skill_bank_path(out_root)
    mem_rows = _load_jsonl(mem_p)
    if not mem_rows:
        return 0
    existing = {
        (r.get("skill_id"), t)
        for r in load_skill_bank(bank_p)
        for t in (r.get("source_tasks") or [])
    }
    n = 0
    for row in mem_rows:
        task = str(row.get("task") or "")
        lesson = row.get("lesson") or {}
        skill = extract_skill(
            task_file=task,
            logdir=str(row.get("logdir") or ""),
            summary=row,
            diagnosis=str(lesson.get("diagnosis") or ""),
            why=str(lesson.get("why") or ""),
            proposal=str(lesson.get("proposal") or ""),
            recovery=str(lesson.get("recovery_prompt_snip") or ""),
        )
        if not skill:
            continue
        key = (skill["skill_id"], os.path.basename(task))
        if key in existing:
            continue
        upsert_skill(bank_p, skill)
        existing.add(key)
        n += 1
    return n


def overlay_promoted_skills(harness: Any, out_root: str) -> List[Dict[str, Any]]:
    """Attach promoted skills onto harness.meta for this task (does not rewrite H0 file)."""
    hydrate_skill_bank(out_root)
    task_cfg = str(getattr(getattr(harness, "runtime", None), "task_config", "") or "")
    ref = parse_task_ref(task_cfg)
    rows = load_skill_bank(skill_bank_path(out_root))
    candidates = [
        s for s in rows
        if not (s.get("task_types") or [])
        or ref["task_type"] in (s.get("task_types") or [])
    ]
    # Unseen catalog entries are explicit hypotheses (n_tasks=0), not priors.
    # Including them gives UCB real alternatives and lets sandbox evidence
    # reject as well as promote knowledge.
    seen_ids = {str(s.get("skill_id") or "") for s in candidates}
    for skill_id, cat in SKILL_CATALOG.items():
        if skill_id in seen_ids or ref["task_type"] not in cat["task_types"]:
            continue
        candidates.append({
            "skill_id": skill_id,
            "triggers": list(cat["triggers"]),
            "task_types": list(cat["task_types"]),
            "template": cat["template"],
            "semantic_actions": list(cat["semantic_actions"]),
            "promoted": False,
            "n_tasks": 0,
            "source_tasks": [],
            "hypothesis": True,
        })
    skills = [s for s in candidates if s.get("promoted")]
    meta = dict(getattr(harness, "meta", None) or {})
    meta["current_target"] = ref["target"]
    meta["current_task_type"] = ref["task_type"]
    meta["prior_skills"] = [
        {
            "skill_id": s.get("skill_id"),
            "triggers": s.get("triggers"),
            "template": s.get("template"),
            "semantic_actions": s.get("semantic_actions"),
            "promoted": True,
            "n_tasks": s.get("n_tasks"),
        }
        for s in skills
    ]
    # Candidate skills are not trusted priors yet. Knowledge probing chooses one
    # arm under sandbox feedback; promotion remains a separate evidence gate.
    meta["knowledge_candidates"] = [
        {
            "skill_id": s.get("skill_id"),
            "triggers": s.get("triggers"),
            "template": s.get("template"),
            "semantic_actions": s.get("semantic_actions"),
            "promoted": bool(s.get("promoted")),
            "n_tasks": s.get("n_tasks"),
            "task_types": s.get("task_types"),
            "source_tasks": s.get("source_tasks"),
        }
        for s in candidates
        if s.get("skill_id") and s.get("template")
    ]
    harness.meta = meta
    return skills


def bind_skill_recovery(
    harness: Any,
    *,
    tags: Optional[Sequence[str]] = None,
    target: str = "",
) -> str:
    """Pick a promoted skill template and bind {target}. Empty if none match."""
    meta = dict(getattr(harness, "meta", None) or {})
    skills = list(meta.get("prior_skills") or [])
    if not skills:
        return ""
    tgt = target or str(meta.get("current_target") or "target")
    tagset = {str(t) for t in (tags or [])}
    picked = None
    for s in skills:
        trig = {str(x) for x in (s.get("triggers") or [])}
        if tagset and trig and tagset & trig:
            picked = s
            break
    if picked is None:
        # stall/fail recover with no attribution: first promoted skill for this task
        picked = skills[0]
    tmpl = str(picked.get("template") or "")
    if not tmpl:
        return ""
    return tmpl.replace("{target}", tgt)


def format_skills_for_prompt(
    rows: Sequence[Dict[str, Any]], *, max_chars: int = 1800
) -> str:
    slim = []
    for r in list(rows)[-12:]:
        slim.append(
            {
                "skill_id": r.get("skill_id"),
                "triggers": r.get("triggers"),
                "template": r.get("template"),
                "promoted": bool(r.get("promoted")),
                "n_tasks": r.get("n_tasks"),
                "holdout_win": r.get("holdout_win"),
                "from": (r.get("source_tasks") or r.get("from_task")),
            }
        )
    if not slim:
        return "(empty)"
    return json.dumps(slim, ensure_ascii=False, default=str)[:max_chars]
