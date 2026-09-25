"""Research evidence layer for Auto-Harness (pre-proposal attribution).

Pipeline upgrade:
  failure → (cluster tags + attribution probes) → evidence → LLM → H'

This is the Researcher experiment layer: distinguish hypotheses *before*
editing the harness. Sandbox remains the sole ACCEPT authority afterward.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Coarse OpenHA / mine-oriented failure modes (not combat-specific).
CLUSTER_DEFS = {
    "zero_progress": "score≈0 / no success signal",
    "stall_heavy": "many recover/stall interventions",
    "probe_heavy": "many probes fired (exploration under failure)",
    "cascade": "cascade_fail marked",
    "near_miss": "partial reward but no success",
    "orientation": "likely facing wrong way / sky / wall (from POV probe)",
    "target_loss": "cannot locate / center target block",
    "execution_gap": "knows what to do but fails to execute (from probe)",
    "instruction_gap": "instruction/recovery text insufficient (from probe)",
}


def infer_failure_tags(
    metrics: Optional[Dict[str, Any]],
    *,
    attribution: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Heuristic cluster tags from last eval (+ optional probe attribution)."""
    tags: List[str] = []
    m = metrics or {}
    sr = float(m.get("success_rate") or 0.0)
    score = float(m.get("score") or 0.0)
    cascade = float(m.get("cascade_rate") or 0.0)
    stats = m.get("runtime_stats") or {}
    n_recover = int(stats.get("recover") or 0)
    n_probe = int(stats.get("probe") or 0)
    # episode-level aggregates if present
    eps = m.get("episodes") or []
    if eps:
        n_recover = max(
            n_recover, sum(int(e.get("n_recover") or 0) for e in eps)
        )
        n_probe = max(n_probe, sum(int(e.get("n_probe") or 0) for e in eps))
        rewards = [float(e.get("reward") or 0.0) for e in eps]
    else:
        rewards = [float(m.get("avg_reward") or 0.0)]

    if sr < 0.05 and score < 0.05:
        tags.append("zero_progress")
    if cascade >= 0.5:
        tags.append("cascade")
    if n_recover >= 3:
        tags.append("stall_heavy")
    if n_probe >= 4:
        tags.append("probe_heavy")
    if sr < 0.5 and max(rewards) > 0.2:
        tags.append("near_miss")

    attr = attribution or {}
    pov = str((attr.get("pov_content") or attr.get("pov") or "")).lower()
    cause = str((attr.get("likely_cause") or attr.get("cause") or "")).lower()
    if any(k in pov for k in ("sky", "wall", "ceiling", "wrong")):
        tags.append("orientation")
    if "target" in cause or "locate" in cause or "center" in str(attr.get("summary") or "").lower():
        tags.append("target_loss")
    if "execution" in cause:
        tags.append("execution_gap")
    if "instruction" in cause or "prompt" in cause:
        tags.append("instruction_gap")

    # unique preserve order
    out: List[str] = []
    for t in tags:
        if t not in out:
            out.append(t)
    return out or ["unspecified_fail"]


def failure_cluster_path(out_root: str) -> str:
    return os.path.join(out_root, "failure_clusters.jsonl")


def update_failure_clusters(
    path: str,
    *,
    task_file: str,
    tags: Sequence[str],
    score: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append one observation; return updated frequency snapshot for top tags."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {
        "task": task_file,
        "tags": list(tags),
        "score": float(score),
        "timestamp": time.time(),
        "meta": meta or {},
    }
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # frequency over recent rows
    counts: Dict[str, int] = {}
    recent: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recent.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        recent = [row]
    for r in recent[-200:]:
        for t in r.get("tags") or []:
            counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    return {
        "n_records": len(recent),
        "tag_counts": dict(ranked[:16]),
        "top_clusters": [
            {
                "tag": t,
                "count": c,
                "meaning": CLUSTER_DEFS.get(t, ""),
            }
            for t, c in ranked[:8]
        ],
        "this_task_tags": list(tags),
    }


def format_cluster_brief(snapshot: Dict[str, Any]) -> str:
    tops = snapshot.get("top_clusters") or []
    if not tops:
        return "(no clusters yet)"
    parts = [
        f"{c.get('tag')}×{c.get('count')}"
        + (f" ({c.get('meaning')})" if c.get("meaning") else "")
        for c in tops[:6]
    ]
    mine = snapshot.get("this_task_tags") or []
    return (
        "Failure distribution (recent): " + "; ".join(parts)
        + f" | this_task={mine}"
    )


ATTRIBUTION_SYSTEM = (
    "You are a scientific experimenter for a frozen Minecraft VLA harness. "
    "You do NOT propose harness edits yet. "
    "Given POV frames and a failure summary, answer attribution probes that "
    "distinguish hypotheses: perception/orientation vs goal selection vs "
    "execution vs weak instruction. "
    "Return ONE JSON object only."
)


def _extract_json(text: str) -> Dict[str, Any]:
    from curriculum.harness_llm_proposer import extract_json_obj

    return extract_json_obj(text)


def run_attribution_probes(
    *,
    metrics: Optional[Dict[str, Any]],
    image_paths: Sequence[str],
    instruction: str = "",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
) -> Tuple[Dict[str, Any], str]:
    """VLM structured probes → attribution evidence (not a harness patch)."""
    from curriculum.harness_llm_proposer import (
        build_user_message,
        failure_summary,
        llm_api_chat_ex,
    )

    summary = failure_summary(metrics)
    user = (
        "Run attribution probes for this failed / weak Minecraft VLA rollout.\n"
        "Do NOT propose harness JSON edits. Only answer probes.\n\n"
        f"Current instruction: {instruction[:240]!r}\n\n"
        f"Failure summary:\n{json.dumps(summary, ensure_ascii=False, default=str)[:2000]}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "pov_content": "sky|wall|target_visible|ui|underground|other",\n'
        '  "target_in_view": true/false,\n'
        '  "likely_cause": "orientation|target_loss|wrong_goal|execution|instruction|unknown",\n'
        '  "hypotheses": [\n'
        '     {"id":"H_orient","claim":"...","status":"supported|rejected|unknown","evidence":"..."},\n'
        '     {"id":"H_goal","claim":"...","status":"supported|rejected|unknown","evidence":"..."},\n'
        '     {"id":"H_exec","claim":"...","status":"supported|rejected|unknown","evidence":"..."}\n'
        "  ],\n"
        '  "action_priority": ["look_down","turn_left","forward","attack"],\n'
        '  "summary": "one sentence attribution",\n'
        '  "recommended_fix": "what kind of prompt/instruction change is justified"\n'
        "}\n"
    )
    paths = [str(p) for p in image_paths if p][:4]
    messages = [
        {"role": "system", "content": ATTRIBUTION_SYSTEM},
        build_user_message(user, paths),
    ]
    reply = llm_api_chat_ex(
        messages=messages,
        model=model,
        max_tokens=2048,
        temperature=float(temperature),
    )
    raw = str(reply.get("text") or "")
    try:
        obj = _extract_json(raw)
    except Exception as e:
        obj = {
            "pov_content": "other",
            "likely_cause": "unknown",
            "summary": f"attribution_parse_failed: {e}",
            "hypotheses": [],
            "raw_error": str(e),
        }
    obj["_meta"] = {
        "finish_reason": reply.get("finish_reason"),
        "truncated": bool(reply.get("truncated")),
        "n_images": len(paths),
        "vision": bool(paths),
    }
    return obj, raw


def build_research_evidence(
    *,
    metrics: Optional[Dict[str, Any]],
    image_paths: Sequence[str],
    instruction: str = "",
    out_root: str = "",
    task_file: str = "",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    enable_vlm_probes: bool = True,
) -> Dict[str, Any]:
    """Full evidence pack: clusters + optional VLM attribution probes."""
    attribution: Dict[str, Any] = {}
    raw = ""
    if enable_vlm_probes and image_paths:
        try:
            attribution, raw = run_attribution_probes(
                metrics=metrics,
                image_paths=image_paths,
                instruction=instruction,
                model=model,
                temperature=temperature,
            )
        except Exception as e:
            attribution = {
                "likely_cause": "unknown",
                "summary": f"attribution_probe_failed: {e}",
                "hypotheses": [],
            }
            raw = str(e)

    tags = infer_failure_tags(metrics, attribution=attribution)
    cluster_snap: Dict[str, Any] = {
        "this_task_tags": tags,
        "top_clusters": [],
        "n_records": 0,
    }
    if out_root:
        cluster_snap = update_failure_clusters(
            failure_cluster_path(out_root),
            task_file=task_file or "unknown",
            tags=tags,
            score=float((metrics or {}).get("score") or 0.0),
            meta={
                "cause": attribution.get("likely_cause"),
                "pov": attribution.get("pov_content"),
            },
        )

    pack = {
        "tags": tags,
        "cluster": cluster_snap,
        "cluster_brief": format_cluster_brief(cluster_snap),
        "attribution": attribution,
        "attribution_raw": raw[:2000],
        "timestamp": time.time(),
        "pipeline": "F→P→Evid→LLM→H'",
    }
    return pack


def format_research_evidence_for_prompt(
    pack: Optional[Dict[str, Any]], *, max_chars: int = 2200
) -> str:
    if not pack:
        return "(no research evidence; propose carefully)"
    attr = pack.get("attribution") or {}
    slim = {
        "cluster_brief": pack.get("cluster_brief"),
        "this_task_tags": pack.get("tags"),
        "pov_content": attr.get("pov_content"),
        "target_in_view": attr.get("target_in_view"),
        "likely_cause": attr.get("likely_cause"),
        "hypotheses": (attr.get("hypotheses") or [])[:4],
        "action_priority": attr.get("action_priority"),
        "attribution_summary": attr.get("summary"),
        "recommended_fix": attr.get("recommended_fix"),
    }
    text = json.dumps(slim, ensure_ascii=False, default=str)
    return text[:max_chars]
