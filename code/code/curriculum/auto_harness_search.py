#!/usr/bin/env python3
"""Training-free Auto Harness Search for frozen VLAs.

  probing + harness + frozen VLA + sandbox
  → mutate editable harness files → score → keep or revert

No GRPO / no SV training / no WM fine-tune. Search only.

Usage:
  python -m curriculum.auto_harness_search --smoke
  python -m curriculum.auto_harness_search --config curriculum/configs/auto_harness.yaml
  python -m curriculum.run_curriculum smoke_auto_harness
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.harness_runtime import HarnessRuntime
from curriculum.harness_schema import HarnessSpec, default_seed_harness


# --------------------------------------------------------------------------- #
# Change manifest (decision observability — AHE-style)
# --------------------------------------------------------------------------- #

@dataclass
class ChangeManifest:
    round_id: int
    component: str
    op: str
    before: Any
    after: Any
    claim: str
    expect_delta: float
    expected_fix: List[str] = field(default_factory=list)
    risk: str = "low"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposeOutcome:
    """One LLM (or random) proposal attempt for funnel accounting."""

    valid: bool
    reason: str = "ok"  # ok | api | parse | schema | apply | unmatched_bank | other
    detail: str = ""
    cand: Any = None
    manifest: Optional[ChangeManifest] = None


def default_eval_seeds(seed: int, k: int) -> List[int]:
    """Paired episode ids: 101, 202, ... (offset by search seed).

    On OpenHA these are **harness / task_id diversity keys**, not independent
    Minecraft world seeds. World seed stays ``task_config["seed"]`` (often 2025)
    so ``/tp`` scenes remain valid. Diversity = probe RNG + warmup/stall
    timing under the same fixed world.
    """
    k = max(1, int(k))
    return [int(seed) + 101 * (i + 1) for i in range(k)]


def default_holdout_seeds(seed: int, k: int) -> List[int]:
    """Disjoint from eval episode ids: 909, 1010, 1111, ... (same caveat)."""
    k = max(1, int(k))
    return [int(seed) + 909 + 101 * i for i in range(k)]


def episode_return(ep: Dict[str, Any]) -> float:
    """Per-seed R used in J(H). Same mix as aggregate score."""
    succ = 1.0 if ep.get("success") else 0.0
    reward = float(ep.get("reward") or 0.0)
    cascade = 1.0 if ep.get("cascade_fail") else 0.0
    return succ + 0.05 * reward / 10.0 - 0.2 * cascade


def _episode_seed(ep: Dict[str, Any]) -> Optional[int]:
    meta = ep.get("meta") or {}
    if meta.get("seed") is None:
        return None
    return int(meta["seed"])


def paired_decision(
    base: Dict[str, Any],
    cand: Dict[str, Any],
    *,
    accept_eps: float = 0.01,
) -> Dict[str, Any]:
    """ACCEPT iff mean_i Δ_i >= eps, with Δ_i = R(H'; s_i) - R(H; s_i).

    H and H' MUST share the same eval seeds. Missing overlap is a protocol bug.
    """
    base_by = {_episode_seed(e): e for e in (base.get("episodes") or [])}
    cand_by = {_episode_seed(e): e for e in (cand.get("episodes") or [])}
    base_by.pop(None, None)
    cand_by.pop(None, None)
    seeds = sorted(set(base_by) & set(cand_by))
    if not seeds:
        raise ValueError("paired_decision: no overlapping eval seeds between H and H'")
    per: List[Dict[str, Any]] = []
    deltas: List[float] = []
    for s in seeds:
        r_h = episode_return(base_by[s])
        r_hp = episode_return(cand_by[s])
        d = r_hp - r_h
        deltas.append(d)
        per.append({
            "seed": s,
            "R_H": r_h,
            "R_Hp": r_hp,
            "delta": d,
            "succ_H": bool(base_by[s].get("success")),
            "succ_Hp": bool(cand_by[s].get("success")),
            "reward_H": float(base_by[s].get("reward") or 0.0),
            "reward_Hp": float(cand_by[s].get("reward") or 0.0),
        })
    mean_delta = float(sum(deltas) / len(deltas))
    n_win = sum(1 for d in deltas if d > 1e-12)
    n_lose = sum(1 for d in deltas if d < -1e-12)
    keep = mean_delta >= float(accept_eps)
    return {
        "keep": keep,
        "mean_delta": mean_delta,
        "n_seeds": len(seeds),
        "n_win": n_win,
        "n_lose": n_lose,
        "n_tie": len(seeds) - n_win - n_lose,
        "per_seed": per,
        "seeds": seeds,
    }


# --------------------------------------------------------------------------- #
# Mutation operators over editable harness components
# --------------------------------------------------------------------------- #

MUTATION_BANK: List[Dict[str, Any]] = [
    {
        "component": "probe",
        "op": "enable",
        "path": ("probe", "enabled"),
        "value": True,
        "claim": "Active probing fills memory for later policy reuse.",
        "expect_delta": 0.05,
        "expected_fix": ["empty_memory", "repeat_fail"],
    },
    {
        "component": "probe",
        "op": "adaptive_ucb",
        "path": ("probe", "selection_strategy"),
        "value": "ucb",
        "claim": "Contextual probe utility adapts actions while transferring a global prior.",
        "expect_delta": 0.04,
        "expected_fix": ["under_explore", "repeat_fail", "negative_probe_value"],
    },
    {
        "component": "probe",
        "op": "more_frequent",
        "path": ("probe", "every_n_steps"),
        "value": 2,
        "claim": "Denser probes harvest more verified experience.",
        "expect_delta": 0.03,
        "expected_fix": ["under_explore"],
    },
    {
        "component": "probe",
        "op": "higher_budget",
        "path": ("probe", "budget_per_episode"),
        "value": 5,
        "claim": "Larger probe budget improves coverage before task push.",
        "expect_delta": 0.02,
        "expected_fix": ["under_explore"],
    },
    {
        "component": "probe",
        "op": "attack_pool",
        "path": ("probe", "action_pool"),
        "value": ["forward", "attack", "forward_attack", "turn_left", "turn_right"],
        "claim": "Mine-oriented probes (attack) fill memory with tree-break actions.",
        "expect_delta": 0.06,
        "expected_fix": ["no_attack", "under_explore"],
    },
    {
        "component": "probe",
        "op": "navigate_then_attack",
        "path": ("probe", "action_pool"),
        "value": ["forward", "forward_attack", "attack", "look_up", "look_down", "turn_left"],
        "claim": "Probe nav+aim+attack covers oak_log approach and chop.",
        "expect_delta": 0.05,
        "expected_fix": ["miss_tree", "under_explore"],
    },
    {
        "component": "probe",
        "op": "prefer_uncertain",
        "path": ("probe", "prefer_uncertain"),
        "value": True,
        "claim": "Uncertainty-biased probes expand coverage of unused actions.",
        "expect_delta": 0.03,
        "expected_fix": ["under_explore"],
    },
    {
        "component": "probe",
        "op": "dense_attack_budget",
        "path": ("probe", "budget_per_episode"),
        "value": 8,
        "claim": "Heavy probe budget before exploit raises verified attack memory.",
        "expect_delta": 0.04,
        "expected_fix": ["under_explore"],
    },
    {
        "component": "memory",
        "op": "enable",
        "path": ("memory", "enabled"),
        "value": True,
        "claim": "Persisting probe results helps later episodes.",
        "expect_delta": 0.08,
        "expected_fix": ["no_transfer"],
    },
    {
        "component": "memory",
        "op": "inject_prompt",
        "path": ("memory", "inject_into_prompt"),
        "value": True,
        "claim": "Injecting memory into context biases VLA toward verified actions.",
        "expect_delta": 0.04,
        "expected_fix": ["ignore_experience"],
    },
    {
        "component": "memory",
        "op": "write_failure",
        "path": ("memory", "write_failure"),
        "value": True,
        "claim": "Recording failures lets verify block repeats.",
        "expect_delta": 0.05,
        "expected_fix": ["repeat_fail"],
    },
    {
        "component": "verify",
        "op": "enable",
        "path": ("verify", "enabled"),
        "value": True,
        "claim": "Pre-execution abstain gate prevents cascade failures.",
        "expect_delta": 0.06,
        "expected_fix": ["cascade", "bad_action"],
    },
    {
        "component": "verify",
        "op": "stricter_abstain",
        "path": ("verify", "abstain_thresh"),
        "value": 0.55,
        "claim": "Lower abstain threshold blocks uncertain actions earlier.",
        "expect_delta": 0.03,
        "expected_fix": ["cascade"],
    },
    {
        "component": "verify",
        "op": "prefer_memory",
        "path": ("verify", "prefer_memory_action"),
        "value": True,
        "claim": "Prefer memory-suggested actions over reactive VLA guesses.",
        "expect_delta": 0.04,
        "expected_fix": ["ignore_experience"],
    },
    {
        "component": "recover",
        "op": "enable",
        "path": ("recover", "enabled"),
        "value": True,
        "claim": "Rollback + replan stops long-horizon cascade.",
        "expect_delta": 0.10,
        "expected_fix": ["cascade", "recovery_gap"],
    },
    {
        "component": "recover",
        "op": "more_retries",
        "path": ("recover", "max_retries"),
        "value": 5,
        "claim": "More recovery attempts raise success under injected faults.",
        "expect_delta": 0.03,
        "expected_fix": ["recovery_gap"],
    },
    {
        "component": "recover",
        "op": "use_checkpoint",
        "path": ("recover", "use_memory_checkpoint"),
        "value": True,
        "claim": "Checkpoint rollback restores safe progress.",
        "expect_delta": 0.05,
        "expected_fix": ["cascade"],
    },
    {
        "component": "runtime",
        "op": "more_steps",
        "path": ("runtime", "max_steps"),
        "value": 48,
        "claim": "Longer horizon allows recover+progress after faults.",
        "expect_delta": 0.02,
        "expected_fix": ["timeout"],
    },
    {
        "component": "runtime",
        "op": "tighter_checkpoint",
        "path": ("runtime", "checkpoint_every"),
        "value": 4,
        "claim": "Denser checkpoints improve rollback quality.",
        "expect_delta": 0.02,
        "expected_fix": ["cascade"],
    },
    # Negative / ablation-style mutations (search may try and revert)
    {
        "component": "recover",
        "op": "disable",
        "path": ("recover", "enabled"),
        "value": False,
        "claim": "Ablation: recover off (expect drop).",
        "expect_delta": -0.10,
        "expected_fix": [],
        "risk": "high",
    },
    {
        "component": "probe",
        "op": "disable",
        "path": ("probe", "enabled"),
        "value": False,
        "claim": "Ablation: probe off (expect weaker memory).",
        "expect_delta": -0.05,
        "expected_fix": [],
        "risk": "high",
    },
]


def _get_path(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for p in path:
        cur = getattr(cur, p) if not isinstance(cur, dict) else cur[p]
    return cur


def _set_path(obj: Any, path: Sequence[str], value: Any) -> None:
    cur = obj
    for p in path[:-1]:
        cur = getattr(cur, p)
    setattr(cur, path[-1], value)


def apply_mutation(harness: HarnessSpec, mut: Dict[str, Any]) -> Tuple[HarnessSpec, ChangeManifest]:
    h = harness.clone(bump_version=True)
    path = tuple(mut["path"])
    before = _get_path(h, path)
    _set_path(h, path, mut["value"])
    # Coupled side-effects so single-component edits are not vacuous
    if mut["component"] == "recover" and mut["op"] == "enable" and mut["value"] is True:
        h.recovery_prompt = (
            "Previous action failed. Roll back to checkpoint, prefer memory-verified "
            "forward progress, avoid the failed action."
        )
        h.recover.use_memory_checkpoint = True
        if h.recover.max_retries < 8:
            h.recover.max_retries = 8
        if h.runtime.checkpoint_every > 6:
            h.runtime.checkpoint_every = 4
        if h.runtime.max_steps < 36:
            h.runtime.max_steps = 36
    if mut["component"] == "memory" and mut["op"] == "enable" and mut["value"] is True:
        h.memory.inject_into_prompt = True
        h.memory.write_failure = True
        h.memory.write_success = True
    if mut["component"] == "verify" and mut["op"] == "enable" and mut["value"] is True:
        h.verify.prefer_memory_action = True
        h.verify.block_on_abstain = True
    if mut["component"] == "probe" and mut["op"] == "enable" and mut["value"] is True:
        h.memory.enabled = True
        h.memory.write_probe = True
    if mut["component"] == "probe" and mut["op"] == "adaptive_ucb":
        h.probe.enabled = True
        h.memory.enabled = True
        h.memory.write_probe = True
    if mut["component"] == "probe" and mut["op"] in (
        "attack_pool", "navigate_then_attack", "dense_attack_budget",
    ):
        h.probe.enabled = True
        h.memory.enabled = True
        h.memory.write_probe = True
        h.memory.inject_into_prompt = True
        if mut["op"] == "dense_attack_budget":
            h.probe.every_n_steps = min(int(h.probe.every_n_steps), 2)
            h.probe.action_pool = [
                "forward", "attack", "forward_attack", "turn_left", "look_up",
            ]
        if mut["op"] in ("attack_pool", "navigate_then_attack"):
            h.probe.budget_per_episode = max(int(h.probe.budget_per_episode), 4)
            h.probe.every_n_steps = min(int(h.probe.every_n_steps), 3)
    manifest = ChangeManifest(
        round_id=h.version,
        component=str(mut["component"]),
        op=str(mut["op"]),
        before=before,
        after=mut["value"],
        claim=str(mut["claim"]),
        expect_delta=float(mut.get("expect_delta", 0.0)),
        expected_fix=list(mut.get("expected_fix", [])),
        risk=str(mut.get("risk", "low")),
    )
    h.meta["last_mutation"] = manifest.to_dict()
    return h, manifest


def degraded_seed() -> HarnessSpec:
    """Weak seed so search has headroom (HELM-gaps open)."""
    h = default_seed_harness()
    h.name = "degraded_seed"
    h.probe.enabled = False
    h.memory.enabled = False
    h.memory.write_failure = False
    h.memory.inject_into_prompt = False
    h.verify.enabled = False
    h.recover.enabled = False
    h.runtime.max_steps = 36
    h.runtime.checkpoint_every = 20
    h.runtime.backend = "stub"
    h.runtime.vla_mode = "stub"
    return h


def strong_seed() -> HarnessSpec:
    h = default_seed_harness()
    h.name = "strong_seed"
    # Probe on with mine-oriented actions (oak_log needs attack)
    h.probe.enabled = True
    h.probe.every_n_steps = 3
    h.probe.budget_per_episode = 4
    h.probe.prefer_uncertain = True
    h.probe.action_pool = ["forward", "attack", "forward_attack", "turn_left", "turn_right"]
    h.memory.enabled = True
    h.memory.write_failure = True
    h.memory.write_probe = True
    h.memory.inject_into_prompt = True
    h.verify.enabled = True
    h.verify.prefer_memory_action = True
    h.recover.enabled = True
    h.recover.max_retries = 3
    h.recover.use_memory_checkpoint = True
    h.runtime.max_steps = 36
    h.runtime.checkpoint_every = 4
    h.runtime.backend = "stub"
    h.runtime.vla_mode = "stub"
    return h


# --------------------------------------------------------------------------- #
# Search loop
# --------------------------------------------------------------------------- #

def _llm_api_propose_json(
    *,
    prompt: str,
    model: str = "gemini-2.5-flash",
    api_key: str = "",
    request_url: str = "https://api.example.com/v1/chat/completions",
) -> Dict[str, Any]:
    """One LLMAPI chat completion → parsed JSON object (harness hypothesis)."""
    from curriculum.gemini_vla import _resolve_app_id

    key = _resolve_app_id(api_key)
    body = {
        "model": model,
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 512,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You propose a single harness mutation for a frozen VLA. "
                    "Return ONLY one JSON object. Never emit game actions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        request_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90.0) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "\n".join(
            str(p.get("text") if isinstance(p, dict) else p) for p in content
        )
    text = str(content).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    raw = m.group(0) if m else text
    return json.loads(raw)


class AutoHarnessSearch:
    """Observability-driven harness evolution; frozen VLA; sandbox as oracle."""

    def __init__(
        self,
        *,
        logdir: str,
        seed_harness: Optional[HarnessSpec] = None,
        n_rounds: int = 10,
        n_eval_episodes: int = 6,
        seed: int = 0,
        accept_eps: float = 0.01,
        mutation_bank: Optional[List[Dict[str, Any]]] = None,
        prefer_components: Optional[List[str]] = None,
        proposer_mode: str = "random",
        proposer_model: str = "gemini-2.5-flash",
        proposer_temperature: float = 0.0,
        proposer_vision: bool = False,
        proposer_max_frames: int = 4,
        eval_seeds: Optional[Sequence[int]] = None,
        holdout_seeds: Optional[Sequence[int]] = None,
        success_memory: Optional[List[Dict[str, Any]]] = None,
        attribution_probes: bool = True,
        actionable_patches: bool = False,
    ):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.n_rounds = int(n_rounds)
        self.n_eval = int(n_eval_episodes)
        self.eval_seeds = [int(s) for s in (eval_seeds or default_eval_seeds(seed, self.n_eval))]
        self.holdout_seeds = [
            int(s) for s in (holdout_seeds or default_holdout_seeds(seed, max(1, min(2, self.n_eval))))
        ]
        overlap = set(self.eval_seeds) & set(self.holdout_seeds)
        if overlap:
            raise ValueError(f"holdout seeds overlap eval seeds: {sorted(overlap)}")
        self.accept_eps = float(accept_eps)
        self.proposer_mode = str(proposer_mode or "random").lower()
        self.proposer_model = str(proposer_model or "gemini-2.5-flash")
        self.proposer_temperature = float(proposer_temperature)
        self.proposer_vision = bool(proposer_vision)
        self.proposer_max_frames = max(1, int(proposer_max_frames))
        self.success_memory = list(success_memory or [])
        self.attribution_probes = bool(attribution_probes)
        self.actionable_patches = bool(actionable_patches)
        bank = list(mutation_bank or MUTATION_BANK)
        prefs = [str(x) for x in (prefer_components or []) if x]
        if prefs:
            pref_set = set(prefs)
            front = [m for m in bank if str(m.get("component")) in pref_set]
            back = [m for m in bank if str(m.get("component")) not in pref_set]
            bank = front + back
        self.bank = bank
        self.prefer_components = prefs
        self.harness = seed_harness or degraded_seed()
        self.history: List[Dict[str, Any]] = []
        self.best_score = -1e9
        self.best_harness = self.harness.clone(bump_version=False)
        self.seed_harness = self.harness.clone(bump_version=False)
        self._shared_vla = None  # lazy: reuse shared hf weights across rounds

    def _get_shared_vla(self, harness: HarnessSpec):
        mode = str(harness.runtime.vla_mode).lower()
        if mode not in ("hf", "gemini", "gemini-flash", "gemini_flash"):
            return None
        if self._shared_vla is not None:
            return self._shared_vla
        from curriculum.harness_runtime import HarnessRuntime
        tmp = HarnessRuntime(harness, seed=self.seed)
        self._shared_vla = tmp.vla
        return self._shared_vla

    def _eval(self, harness: HarnessSpec, tag: str) -> Dict[str, Any]:
        """J(H) on the fixed eval_seeds. Independent R(H; s_i) per seed."""
        vla = self._get_shared_vla(harness)
        rt = HarnessRuntime(harness, seed=self.eval_seeds[0], vla=vla)
        metrics = rt.evaluate_on_seeds(self.eval_seeds, persist_memory=False)
        metrics["tag"] = tag
        hpath = os.path.join(self.logdir, f"harness_{tag}.json")
        harness.save(hpath)
        mpath = os.path.join(self.logdir, f"memory_{tag}.json")
        rt.memory.save(mpath)
        with open(os.path.join(self.logdir, f"metrics_{tag}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(
            f"[auto_harness] eval tag={tag} seeds={self.eval_seeds} "
            f"score={metrics['score']:.4f} succ={metrics['success_rate']:.3f}",
            flush=True,
        )
        return metrics

    def _pick_mutation(self, tried: set) -> Optional[Dict[str, Any]]:
        cands = []
        for m in self.bank:
            key = (m["component"], m["op"], str(m["value"]))
            if key in tried:
                continue
            # skip no-ops
            path = tuple(m["path"])
            try:
                cur = _get_path(self.harness, path)
            except Exception:
                continue
            if cur == m["value"]:
                continue
            cands.append(m)
        if not cands:
            return None
        # Prefer configured components (e.g. probe-first search)
        if self.prefer_components:
            pref = [m for m in cands if str(m.get("component")) in set(self.prefer_components)]
            if pref:
                cands = pref
        # prefer high-impact positive mutations first
        pos = [m for m in cands if float(m.get("expect_delta", 0)) > 0]
        pool = pos or cands
        pool = sorted(pool, key=lambda m: -float(m.get("expect_delta", 0)))
        # weighted toward top-3 impact
        top = pool[: max(3, min(5, len(pool)))]
        return self.rng.choice(top)

    def _pick_mutation_llm(
        self,
        tried: set,
        last_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """LLMAPI Gemini proposes one bank mutation from failure / score context."""
        cands = []
        for m in self.bank:
            key = (m["component"], m["op"], str(m["value"]))
            if key in tried:
                continue
            try:
                cur = _get_path(self.harness, tuple(m["path"]))
            except Exception:
                continue
            if cur == m["value"]:
                continue
            cands.append(m)
        if not cands:
            return None
        catalog = [
            {
                "component": m["component"],
                "op": m["op"],
                "claim": m.get("claim"),
                "expect_delta": m.get("expect_delta"),
            }
            for m in cands
        ]
        hcompact = {
            "probe": self.harness.probe.__dict__,
            "memory": {
                k: getattr(self.harness.memory, k)
                for k in ("enabled", "inject_into_prompt", "write_failure", "write_probe")
            },
            "verify": self.harness.verify.__dict__,
            "recover": self.harness.recover.__dict__,
            "runtime": {
                k: getattr(self.harness.runtime, k)
                for k in ("max_steps", "action_chunk_len", "instruction")
            },
        }
        slim_metrics: Dict[str, Any] = {}
        if last_metrics:
            slim_metrics = {
                k: last_metrics.get(k)
                for k in (
                    "score", "success_rate", "avg_reward", "cascade_rate",
                    "avg_steps", "tag",
                )
            }
            eps = last_metrics.get("episodes") or []
            slim_metrics["episodes"] = [
                {
                    "success": e.get("success"),
                    "steps": e.get("steps"),
                    "reward": e.get("reward"),
                    "n_probe": e.get("n_probe"),
                    "n_verify_block": e.get("n_verify_block"),
                    "n_recover": e.get("n_recover"),
                    "cascade_fail": e.get("cascade_fail"),
                }
                for e in eps[:4]
            ]
        prompt = (
            "Frozen VLA + MineStudio sandbox. Propose ONE harness mutation.\n\n"
            f"Current harness:\n{json.dumps(hcompact, ensure_ascii=False, default=str)[:2500]}\n\n"
            f"Last eval:\n{json.dumps(slim_metrics, ensure_ascii=False, default=str)[:2000]}\n\n"
            f"Available mutations:\n{json.dumps(catalog, ensure_ascii=False)[:4000]}\n\n"
            "Return JSON: {\"component\":..., \"op\":..., \"reason\":...}\n"
            "Pick the edit most likely to raise success_rate. Minimal change."
        )
        try:
            obj = _llm_api_propose_json(prompt=prompt, model=self.proposer_model)
        except Exception as e:
            print(f"[auto_harness] LLM propose failed ({e}); fallback random", flush=True)
            return self._pick_mutation(tried)
        comp = str(obj.get("component") or "")
        op = str(obj.get("op") or "")
        for m in cands:
            if str(m["component"]) == comp and str(m["op"]) == op:
                print(
                    f"[auto_harness] LLM pick {comp}.{op} "
                    f"reason={str(obj.get('reason') or '')[:80]}",
                    flush=True,
                )
                return m
        print(f"[auto_harness] LLM pick unmatched {comp}.{op}; fallback random", flush=True)
        return self._pick_mutation(tried)

    def _propose_harness_llm(
        self,
        last_metrics: Optional[Dict[str, Any]],
        round_id: int,
    ) -> ProposeOutcome:
        """Full API proposer: structured patch, not random knob bank.

        Research pipeline: failure → attribution probes / clusters → evidence → H'.
        Invalid outcomes are first-class (funnel), not silent skips.
        """
        from curriculum.harness_funnel import classify_proposer_error
        from curriculum.harness_llm_proposer import (
            apply_patch,
            collect_eval_frames,
            propose_patch,
        )
        from curriculum.harness_research_evidence import build_research_evidence

        image_paths: List[str] = []
        if self.proposer_vision:
            img_root = str(
                getattr(self.harness.runtime, "img_save_dir", "") or ""
            ) or os.path.join(self.logdir, "images")
            image_paths = collect_eval_frames(
                img_root, max_frames=self.proposer_max_frames
            )
            print(
                f"[auto_harness] VLM proposer frames={len(image_paths)} "
                f"root={img_root}",
                flush=True,
            )

        out_root = os.path.dirname(os.path.abspath(self.logdir))
        task_file = os.path.basename(self.logdir).replace("search_", "") + ".json"
        instr = str(getattr(self.harness.runtime, "instruction", "") or "")
        research_evidence: Dict[str, Any] = {}
        if self.attribution_probes:
            print(
                f"[auto_harness] research evidence: attribution_probes=True "
                f"frames={len(image_paths)}",
                flush=True,
            )
            research_evidence = build_research_evidence(
                metrics=last_metrics,
                image_paths=image_paths,
                instruction=instr,
                out_root=out_root,
                task_file=task_file,
                model=self.proposer_model,
                temperature=float(self.proposer_temperature),
                enable_vlm_probes=bool(image_paths),
            )
            with open(
                os.path.join(self.logdir, f"attribution_r{round_id}.json"),
                "w", encoding="utf-8",
            ) as f:
                json.dump(research_evidence, f, indent=2, ensure_ascii=False, default=str)
            tags = research_evidence.get("tags") or []
            cause = (research_evidence.get("attribution") or {}).get("likely_cause")
            print(
                f"[auto_harness] evidence tags={tags} cause={cause} "
                f"cluster={research_evidence.get('cluster_brief')}",
                flush=True,
            )

        try:
            patch, raw, meta = propose_patch(
                self.harness,
                last_metrics,
                model=self.proposer_model,
                history=self.history,
                success_memory=self.success_memory,
                research_evidence=research_evidence or None,
                temperature=float(self.proposer_temperature),
                image_paths=image_paths or None,
                actionable_patches=bool(self.actionable_patches),
            )
        except Exception as e:
            reason = classify_proposer_error(e)
            print(
                f"[auto_harness] proposer_invalid reason={reason} ({e})",
                flush=True,
            )
            return ProposeOutcome(valid=False, reason=reason, detail=str(e)[:400])
        with open(os.path.join(self.logdir, f"llm_raw_r{round_id}.txt"), "w") as f:
            f.write(raw)
        with open(os.path.join(self.logdir, f"llm_patch_r{round_id}.json"), "w") as f:
            json.dump(patch, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(self.logdir, f"llm_meta_r{round_id}.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        if meta.get("vision"):
            print(
                f"[auto_harness] proposer vision=True n_images={meta.get('n_images')}",
                flush=True,
            )
        if meta.get("truncated") or meta.get("repaired"):
            print(
                f"[auto_harness] proposer finish={meta.get('finish_reason')} "
                f"truncated={meta.get('truncated')} repaired={meta.get('repaired')} "
                f"attempt={meta.get('attempt')}",
                flush=True,
            )

        if not patch.get("edits") and patch.get("component") and patch.get("op"):
            mut = None
            for m in self.bank:
                if str(m["component"]) == str(patch["component"]) and str(m["op"]) == str(patch["op"]):
                    mut = m
                    break
            if mut is None:
                print("[auto_harness] LLM bank-style pick unmatched", flush=True)
                return ProposeOutcome(
                    valid=False,
                    reason="unmatched_bank",
                    detail=f"{patch.get('component')}.{patch.get('op')}",
                )
            cand, manifest = apply_mutation(self.harness, mut)
            manifest.claim = str(patch.get("reason") or patch.get("claim") or manifest.claim)
            return ProposeOutcome(valid=True, cand=cand, manifest=manifest)

        try:
            cand, applied = apply_patch(self.harness, patch)
        except Exception as e:
            reason = classify_proposer_error(e)
            if reason == "other":
                reason = "apply"
            print(f"[auto_harness] LLM patch rejected reason={reason} ({e})", flush=True)
            return ProposeOutcome(valid=False, reason=reason, detail=str(e)[:400])
        claim = str(
            patch.get("claim") or patch.get("proposal") or patch.get("reason") or "llm harness patch"
        )
        diagnosis = str(patch.get("diagnosis") or "")[:240]
        why = str(patch.get("why") or "")[:240]
        proposal = str(patch.get("proposal") or "")[:320]
        manifest = ChangeManifest(
            round_id=int(round_id),
            component="harness",
            op="llm_patch",
            before=[{"path": a["path"], "value": a["before"]} for a in applied],
            after=[{"path": a["path"], "value": a["after"]} for a in applied],
            claim=claim,
            expect_delta=float(patch.get("expect_delta") or 0.05),
            expected_fix=["llm_hypothesis"],
            risk="low",
        )
        md = manifest.to_dict()
        md["diagnosis"] = diagnosis
        md["why"] = why
        md["proposal"] = proposal
        if research_evidence:
            md["research_tags"] = research_evidence.get("tags")
            md["attribution_cause"] = (research_evidence.get("attribution") or {}).get(
                "likely_cause"
            )
        cand.meta["last_mutation"] = md
        cand.meta["last_diagnosis"] = diagnosis
        cand.meta["last_why"] = why
        cand.meta["last_proposal"] = proposal
        if research_evidence:
            cand.meta["last_research_evidence"] = {
                "tags": research_evidence.get("tags"),
                "cause": (research_evidence.get("attribution") or {}).get("likely_cause"),
                "summary": (research_evidence.get("attribution") or {}).get("summary"),
            }
        # Open a Hypothesis Registry trial (closed after sandbox paired eval).
        from curriculum.harness_hypothesis_registry import open_hypothesis

        cand.meta["open_hypothesis"] = open_hypothesis(
            round_id=int(round_id),
            task_file=task_file,
            diagnosis=diagnosis,
            why=why,
            proposal=proposal,
            claim=claim,
            suspected_cause=str(
                (research_evidence.get("attribution") or {}).get("likely_cause") or ""
            ),
            failure_clusters=list(research_evidence.get("tags") or []),
            evidence={
                "cluster_brief": research_evidence.get("cluster_brief"),
                "attribution_summary": (research_evidence.get("attribution") or {}).get(
                    "summary"
                ),
                "hypotheses": (research_evidence.get("attribution") or {}).get(
                    "hypotheses"
                ),
            }
            if research_evidence
            else {},
            proposed_change=[{"path": a["path"], "after": a["after"]} for a in applied],
            predicted_effect=float(patch.get("expect_delta") or 0.05),
            predicted_regression=str(patch.get("predicted_regression") or "")[:240],
            parent_harness_fp=self.harness.fingerprint(),
            candidate_harness_fp=cand.fingerprint(),
        )
        print(
            f"[auto_harness] LLM patch n_edits={len(applied)} claim={claim[:80]}",
            flush=True,
        )
        if diagnosis or proposal:
            print(
                f"[auto_harness] diagnosis={diagnosis[:100]} | "
                f"proposal={proposal[:100]}",
                flush=True,
            )
        return ProposeOutcome(valid=True, cand=cand, manifest=manifest)

    def run(self) -> Dict[str, Any]:
        self.harness.save(os.path.join(self.logdir, "harness_seed.json"))
        print(
            f"[auto_harness] paired eval_seeds={self.eval_seeds} "
            f"holdout_seeds={self.holdout_seeds} accept_eps={self.accept_eps} "
            f"actionable_patches={self.actionable_patches}",
            flush=True,
        )
        base = self._eval(self.harness, "seed")
        self.best_score = float(base["score"])
        self.best_harness = self.harness.clone(bump_version=False)
        baseline = base
        self.history.append({
            "round": 0,
            "accepted": True,
            "valid": None,
            "score": self.best_score,
            "success_rate": base["success_rate"],
            "manifest": None,
            "metrics": base,
            "eval_seeds": list(self.eval_seeds),
        })
        print(
            f"[auto_harness] seed score={self.best_score:.4f} "
            f"succ={base['success_rate']:.3f} cascade={base['cascade_rate']:.3f}",
            flush=True,
        )

        tried: set = set()
        accepted = 0
        reverted = 0
        proposer_failed = 0
        n_propose = 0
        n_valid = 0
        invalid_reasons: Dict[str, int] = {}
        last_metrics = base

        for r in range(1, self.n_rounds + 1):
            n_propose += 1
            if self.proposer_mode in ("llm", "api", "llm_api"):
                outcome = self._propose_harness_llm(last_metrics, r)
                if not outcome.valid:
                    proposer_failed += 1
                    invalid_reasons[outcome.reason] = invalid_reasons.get(outcome.reason, 0) + 1
                    self.history.append({
                        "round": r,
                        "accepted": False,
                        "valid": False,
                        "invalid_reason": outcome.reason,
                        "invalid_detail": outcome.detail,
                    })
                    with open(os.path.join(self.logdir, "history.json"), "w") as f:
                        json.dump(self.history, f, indent=2, default=str)
                    print(
                        f"[auto_harness] round={r} invalid_patch reason={outcome.reason} "
                        f"(not harness_failed), skip",
                        flush=True,
                    )
                    continue
                cand, manifest = outcome.cand, outcome.manifest
            else:
                mut = self._pick_mutation(tried)
                if mut is None:
                    n_propose -= 1
                    print(f"[auto_harness] round={r} no more mutations", flush=True)
                    break
                key = (mut["component"], mut["op"], str(mut["value"]))
                tried.add(key)
                cand, manifest = apply_mutation(self.harness, mut)
            n_valid += 1
            with open(os.path.join(self.logdir, f"manifest_r{r}.json"), "w") as f:
                md = manifest.to_dict()
                # Preserve diagnosis/proposal if attached on candidate meta.
                if isinstance(getattr(cand, "meta", None), dict):
                    for k in ("diagnosis", "why", "proposal"):
                        v = cand.meta.get(f"last_{k}") or cand.meta.get(k)
                        if v:
                            md[k] = v
                    lm = cand.meta.get("last_mutation") or {}
                    if isinstance(lm, dict):
                        for k in ("diagnosis", "why", "proposal"):
                            if lm.get(k) and not md.get(k):
                                md[k] = lm[k]
                json.dump(md, f, indent=2, default=str)

            metrics = self._eval(cand, f"r{r}")
            last_metrics = dict(metrics)
            last_metrics["baseline_score"] = float(baseline["score"])
            last_metrics["baseline_success_rate"] = float(baseline["success_rate"])
            paired = paired_decision(baseline, metrics, accept_eps=self.accept_eps)
            with open(os.path.join(self.logdir, f"paired_r{r}.json"), "w") as f:
                json.dump(paired, f, indent=2)
            keep = bool(paired["keep"])
            # never accept clearly harmful mutations that claim positive gain
            if manifest.expect_delta > 0 and paired["mean_delta"] < -self.accept_eps:
                keep = False
            # ablation / high-risk edits are for falsification only — never promote
            if str(manifest.risk) == "high" or float(manifest.expect_delta) < 0:
                keep = False
            delta = float(paired["mean_delta"])
            score = float(metrics["score"])

            man_dict = manifest.to_dict()
            if isinstance(getattr(cand, "meta", None), dict):
                for k in ("diagnosis", "why", "proposal"):
                    v = cand.meta.get(f"last_{k}")
                    if v:
                        man_dict[k] = v
            record = {
                "round": r,
                "accepted": keep,
                "valid": True,
                "score": score,
                "delta": delta,
                "success_rate": metrics["success_rate"],
                "cascade_rate": metrics["cascade_rate"],
                "manifest": man_dict,
                "paired": paired,
                "metrics": {
                    k: metrics[k]
                    for k in (
                        "score", "success_rate", "avg_reward", "cascade_rate",
                        "avg_steps", "memory_items", "harness_fp", "eval_seeds",
                    )
                    if k in metrics
                },
            }
            self.history.append(record)

            per_txt = " ".join(
                f"s{p['seed']}:{p['delta']:+.3f}" for p in paired["per_seed"]
            )
            if keep:
                accepted += 1
                self.harness = cand
                self.best_score = score
                self.best_harness = cand.clone(bump_version=False)
                baseline = metrics
                print(
                    f"[auto_harness] r{r} ACCEPT {manifest.component}.{manifest.op} "
                    f"meanΔ={delta:+.4f} score={score:.4f} "
                    f"win={paired['n_win']}/{paired['n_seeds']} {per_txt} "
                    f"claim={manifest.claim[:60]}",
                    flush=True,
                )
            else:
                reverted += 1
                print(
                    f"[auto_harness] r{r} REVERT {manifest.component}.{manifest.op} "
                    f"meanΔ={delta:+.4f} score={score:.4f} "
                    f"win={paired['n_win']}/{paired['n_seeds']} {per_txt} (falsified)",
                    flush=True,
                )

            # Close Hypothesis Registry trial (prediction vs sandbox actual).
            try:
                from curriculum.harness_hypothesis_registry import (
                    append_hypothesis,
                    close_hypothesis,
                    hypothesis_registry_path,
                )

                open_h = None
                if isinstance(getattr(cand, "meta", None), dict):
                    open_h = cand.meta.get("open_hypothesis")
                if isinstance(open_h, dict):
                    closed = close_hypothesis(
                        open_h,
                        accepted=keep,
                        actual_effect=delta,
                        actual_success_rate=float(metrics.get("success_rate") or 0.0),
                    )
                    out_root = os.path.dirname(os.path.abspath(self.logdir))
                    append_hypothesis(hypothesis_registry_path(out_root), closed)
                    with open(
                        os.path.join(self.logdir, f"hypothesis_r{r}.json"), "w"
                    ) as f:
                        json.dump(closed, f, indent=2, ensure_ascii=False, default=str)
                    print(
                        f"[auto_harness] hypothesis {closed.get('hypothesis_id')} "
                        f"accepted={keep} pred={closed.get('predicted_effect')} "
                        f"actualΔ={delta:+.4f} "
                        f"pred_ok={closed.get('prediction_correct')}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[auto_harness] hypothesis_registry_failed ({e})", flush=True)

            with open(os.path.join(self.logdir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2, default=str)

        self.best_harness.save(os.path.join(self.logdir, "harness_best.json"))
        vla = self._get_shared_vla(self.best_harness)
        hold = HarnessRuntime(
            self.best_harness, seed=self.holdout_seeds[0], vla=vla
        ).evaluate_on_seeds(self.holdout_seeds, persist_memory=False)
        with open(os.path.join(self.logdir, "metrics_holdout_best.json"), "w") as f:
            json.dump(hold, f, indent=2, default=str)

        hold_seed = None
        hold_paired = None
        holdout_win = False
        holdout_legacy = False
        if accepted > 0:
            hold_seed = HarnessRuntime(
                self.seed_harness, seed=self.holdout_seeds[0], vla=vla
            ).evaluate_on_seeds(self.holdout_seeds, persist_memory=False)
            with open(os.path.join(self.logdir, "metrics_holdout_seed.json"), "w") as f:
                json.dump(hold_seed, f, indent=2, default=str)
            hold_paired = paired_decision(hold_seed, hold, accept_eps=self.accept_eps)
            holdout_win = bool(hold_paired.get("keep"))
            with open(os.path.join(self.logdir, "paired_holdout.json"), "w") as f:
                json.dump(hold_paired, f, indent=2, default=str)
        else:
            # H* is still H0; no ACCEPT → holdout win is false by definition.
            hold_seed = hold
            holdout_win = False

        # Paired ACCEPT/REVERT evaluation is isolated. Only after selection,
        # Commit exactly one best-policy rollout after paired selection. Both
        # action and knowledge bandits stay isolated during ACCEPT/REVERT.
        adaptive_probe_commit = None
        action_learning = (
            str(getattr(self.best_harness.probe, "selection_strategy", "coverage")).lower()
            == "ucb"
            and bool(str(getattr(self.best_harness.probe, "bandit_state_path", "") or ""))
        )
        knowledge_learning = (
            bool(getattr(self.best_harness.knowledge_probe, "enabled", False))
            and str(
                getattr(self.best_harness.knowledge_probe, "selection_strategy", "ucb")
            ).lower() == "ucb"
            and bool(
                str(
                    getattr(
                        self.best_harness.knowledge_probe, "bandit_state_path", ""
                    ) or ""
                )
            )
        )
        if action_learning or knowledge_learning:
            adaptive_probe_commit = HarnessRuntime(
                self.best_harness, seed=self.eval_seeds[0], vla=vla
            ).evaluate_on_seeds([self.eval_seeds[0]], persist_memory=True)
            with open(os.path.join(self.logdir, "metrics_probe_learning.json"), "w") as f:
                json.dump(adaptive_probe_commit, f, indent=2, default=str)
            print(
                "[auto_harness] committed adaptive probe credit "
                f"action_state={self.best_harness.probe.bandit_state_path} "
                f"knowledge_state={self.best_harness.knowledge_probe.bandit_state_path}",
                flush=True,
            )

        seed_sr = float(base.get("success_rate") or 0.0)
        best_sr = seed_sr
        if accepted > 0:
            for rec in reversed(self.history):
                if rec.get("accepted") and rec.get("valid"):
                    best_sr = float(rec.get("success_rate") or best_sr)
                    break
        delta_sr = float(best_sr) - float(seed_sr)

        from curriculum.harness_funnel import funnel_from_counts, format_funnel

        funnel = funnel_from_counts(
            n_propose=n_propose,
            n_valid=n_valid,
            n_accept=accepted,
            n_task_accept=1 if accepted > 0 else 0,
            n_holdout_win=1 if holdout_win else 0,
            n_tasks=1,
            sum_delta_sr=delta_sr,
            invalid_reasons=invalid_reasons,
        )
        funnel["seed_success_rate"] = seed_sr
        funnel["best_success_rate"] = best_sr
        funnel["delta_success_rate"] = delta_sr
        funnel["holdout_win"] = holdout_win
        funnel["holdout_legacy"] = holdout_legacy
        with open(os.path.join(self.logdir, "funnel.json"), "w") as f:
            json.dump(funnel, f, indent=2)

        summary = {
            "mechanism_id": "training_free_harness_search",
            "seed_score": self.history[0]["score"],
            "best_score": self.best_score,
            "seed_success_rate": seed_sr,
            "best_success_rate": best_sr,
            "delta_success_rate": delta_sr,
            "holdout_score": hold["score"],
            "holdout_success_rate": hold["success_rate"],
            "holdout_seed_success_rate": (
                None if hold_seed is None else hold_seed.get("success_rate")
            ),
            "holdout_win": holdout_win,
            "holdout_paired": hold_paired,
            "eval_seeds": list(self.eval_seeds),
            "holdout_seeds": list(self.holdout_seeds),
            "accepted": accepted,
            "reverted": reverted,
            "proposer_failed": proposer_failed,
            "n_propose": n_propose,
            "n_valid": n_valid,
            "invalid_reasons": invalid_reasons,
            "funnel": funnel,
            "rounds": len(self.history) - 1,
            "adaptive_probe_committed": adaptive_probe_commit is not None,
            "action_probe_committed": bool(adaptive_probe_commit is not None and action_learning),
            "knowledge_probe_committed": bool(
                adaptive_probe_commit is not None and knowledge_learning
            ),
            "probe_bandit_state_path": str(
                getattr(self.best_harness.probe, "bandit_state_path", "") or ""
            ),
            "knowledge_bandit_state_path": str(
                getattr(
                    self.best_harness.knowledge_probe, "bandit_state_path", ""
                ) or ""
            ),
            "best_harness_fp": self.best_harness.fingerprint(),
            "logdir": self.logdir,
            "warm_start_from": (self.harness.meta or {}).get("warm_start_from")
            or (self.best_harness.meta or {}).get("warm_start_from"),
            "effective": bool(
                accepted > 0
                or float(self.best_score) > float(self.history[0]["score"]) + 1e-9
            ),
        }
        with open(os.path.join(self.logdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(
            f"[auto_harness] done best={self.best_score:.4f} "
            f"holdout_succ={hold['success_rate']:.3f} "
            f"accepted={accepted} reverted={reverted} proposer_failed={proposer_failed}",
            flush=True,
        )
        print(format_funnel(funnel), flush=True)
        return summary


def run_from_config(cfg_path: str) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        d = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        with open(cfg_path) as f:
            d = json.load(f)

    logdir = str(d.get("logdir", "curriculum/outputs/auto_harness"))
    seed = int(d.get("seed", 0))
    n_rounds = int(d.get("n_rounds", 10))
    n_eval = int(d.get("n_eval_episodes", 6))
    eval_seeds = d.get("eval_seeds")
    holdout_seeds = d.get("holdout_seeds")
    if eval_seeds is not None:
        eval_seeds = [int(x) for x in list(eval_seeds)]
        n_eval = len(eval_seeds)
    if holdout_seeds is not None:
        holdout_seeds = [int(x) for x in list(holdout_seeds)]
    start = str(d.get("start", "degraded"))  # degraded | strong | path
    warm_from_file = False
    if start == "strong":
        h = strong_seed()
    elif start == "degraded":
        h = degraded_seed()
    elif os.path.isfile(start):
        h = HarnessSpec.load(start)
        warm_from_file = True
        h.name = f"warm_{os.path.basename(start)}"
        print(f"[auto_harness] warm-start from effective prior: {start}", flush=True)
    else:
        h = degraded_seed()

    if d.get("backend"):
        h.runtime.backend = str(d["backend"])
    if d.get("vla_mode"):
        h.runtime.vla_mode = str(d["vla_mode"])
    for k in (
        "ticks_per_action", "task_config", "img_save_dir",
        "success_reward_thresh", "soft_rollback", "max_steps",
        "checkpoint_every", "instruction", "vla_model_path",
        "vla_device", "vla_dtype", "action_chunk_len",
        "gemini_api_key", "gemini_temperature",
        "vla_temperature", "vla_do_sample", "vla_protocol",
    ):
        if k not in d or d[k] is None or d[k] == "":
            continue
        if hasattr(h.runtime, k):
            typ = type(getattr(h.runtime, k))
            setattr(h.runtime, k, typ(d[k]) if typ is not bool else bool(d[k]))
    # Convenience: temperature<=0 ⇒ greedy
    if "vla_temperature" in d and d["vla_temperature"] is not None:
        if float(d["vla_temperature"]) <= 0:
            h.runtime.vla_do_sample = False
            h.runtime.vla_temperature = 0.0
    if "vla_do_sample" in d and d["vla_do_sample"] is not None:
        h.runtime.vla_do_sample = bool(d["vla_do_sample"])

    # When warm-starting from an accepted prior, keep its probe/recover/memory body.
    # Only re-bind task/runtime so evolution actually transfers across tasks.
    preserve = bool(d.get("preserve_harness_body", warm_from_file))
    if not preserve:
        # Optional probe overrides for deeper probe-direction search
        if d.get("probe_action_pool"):
            h.probe.action_pool = [str(x) for x in list(d["probe_action_pool"])]
            h.probe.enabled = True
        if d.get("probe_budget") is not None:
            h.probe.budget_per_episode = int(d["probe_budget"])
        if d.get("probe_every_n") is not None:
            h.probe.every_n_steps = int(d["probe_every_n"])
        if d.get("recover_max_retries") is not None:
            h.recover.max_retries = int(d["recover_max_retries"])
        if "verify_enabled" in d:
            h.verify.enabled = bool(d["verify_enabled"])
        if "prefer_memory_action" in d:
            h.verify.prefer_memory_action = bool(d["prefer_memory_action"])
        if "probe_enabled" in d:
            h.probe.enabled = bool(d["probe_enabled"])
        if "probe_periodic" in d:
            h.probe.probe_periodic = bool(d["probe_periodic"])
        if "probe_on_fail" in d:
            h.probe.probe_on_fail = bool(d["probe_on_fail"])
        if "probe_on_stall" in d:
            h.probe.probe_on_stall = bool(d["probe_on_stall"])
        if d.get("probe_stall_steps") is not None:
            h.probe.stall_steps = int(d["probe_stall_steps"])
        if d.get("warmup_probes") is not None:
            h.probe.warmup_probes = int(d["warmup_probes"])
        probe_overrides = {
            "probe_selection_strategy": "selection_strategy",
            "probe_ucb_c": "ucb_c",
            "probe_transfer_weight": "transfer_weight",
            "probe_downstream_success_reward": "downstream_success_reward",
            "probe_downstream_discount": "downstream_discount",
            "probe_bandit_state_path": "bandit_state_path",
        }
        for config_key, attr in probe_overrides.items():
            if config_key in d and d[config_key] is not None:
                old_value = getattr(h.probe, attr)
                value_type = type(old_value)
                setattr(h.probe, attr, value_type(d[config_key]))
        knowledge_probe_overrides = {
            "knowledge_probe_enabled": "enabled",
            "knowledge_probe_selection_strategy": "selection_strategy",
            "knowledge_probe_ucb_c": "ucb_c",
            "knowledge_probe_transfer_weight": "transfer_weight",
            "knowledge_probe_downstream_success_reward": "downstream_success_reward",
            "knowledge_probe_downstream_discount": "downstream_discount",
            "knowledge_probe_bandit_state_path": "bandit_state_path",
        }
        for config_key, attr in knowledge_probe_overrides.items():
            if config_key in d and d[config_key] is not None:
                old_value = getattr(h.knowledge_probe, attr)
                value_type = type(old_value)
                setattr(
                    h.knowledge_probe, attr,
                    value_type(d[config_key]) if value_type is not bool else bool(d[config_key]),
                )
        if "recover_enabled" in d:
            h.recover.enabled = bool(d["recover_enabled"])
        if "recover_on_stall" in d:
            h.recover.recover_on_stall = bool(d["recover_on_stall"])
        if d.get("recover_stall_steps") is not None:
            h.recover.stall_steps = int(d["recover_stall_steps"])
        if "memory_enabled" in d:
            h.memory.enabled = bool(d["memory_enabled"])
            if not h.memory.enabled:
                h.memory.inject_into_prompt = False
                h.memory.write_success = False
                h.memory.write_failure = False
                h.memory.write_probe = False
    else:
        print(
            "[auto_harness] preserve_harness_body=True "
            "(keeping prior probe/recover/memory; only task/runtime rebound)",
            flush=True,
        )
        # still allow recover_max_retries clamp if provided
        if d.get("recover_max_retries") is not None:
            h.recover.max_retries = int(d["recover_max_retries"])

    if warm_from_file:
        h.meta = dict(h.meta or {})
        h.meta["warm_start_from"] = os.path.abspath(start)

    try:
        from curriculum.harness_skill_bank import overlay_promoted_skills

        overlay_promoted_skills(h, os.path.dirname(os.path.abspath(logdir)))
    except Exception as e:
        print(f"[auto_harness] skill overlay skipped: {e}", flush=True)

    prefer_components = None
    bank = None
    if d.get("prefer_components"):
        prefer_components = [str(x) for x in list(d["prefer_components"])]
    if d.get("mutation_components"):
        allow = set(str(x) for x in list(d["mutation_components"]))
        bank = [m for m in MUTATION_BANK if str(m.get("component")) in allow]
        if not prefer_components:
            prefer_components = list(allow)

    if str(h.runtime.backend) == "minestudio" and str(h.runtime.vla_mode).lower() in (
        "hf", "gemini", "gemini-flash", "gemini_flash",
    ):
        print(
            f"[auto_harness] live VLA={h.runtime.vla_mode} + MineStudio "
            f"task={h.runtime.task_config!r} prefer={prefer_components}",
            flush=True,
        )
    searcher = AutoHarnessSearch(
        logdir=logdir,
        seed_harness=h,
        n_rounds=n_rounds,
        n_eval_episodes=n_eval,
        seed=seed,
        accept_eps=float(d.get("accept_eps", 0.01)),
        mutation_bank=bank,
        prefer_components=prefer_components,
        proposer_mode=str(d.get("proposer_mode") or "random"),
        proposer_model=str(d.get("proposer_model") or "gemini-2.5-flash"),
        proposer_temperature=float(d.get("proposer_temperature", 0.0)),
        proposer_vision=bool(d.get("proposer_vision", False)),
        proposer_max_frames=int(d.get("proposer_max_frames", 4) or 4),
        eval_seeds=eval_seeds,
        holdout_seeds=holdout_seeds,
        success_memory=(
            __import__(
                "curriculum.harness_success_memory", fromlist=["load_success_memory"]
            ).load_success_memory(str(d["success_memory_jsonl"]))
            if d.get("success_memory_jsonl")
            else []
        ),
        attribution_probes=bool(d.get("attribution_probes", True)),
        actionable_patches=bool(d.get("actionable_patches", False)),
    )
    return searcher.run()


def ping_minestudio(max_steps: int = 2, vla_mode: str = "stub") -> Dict[str, Any]:
    """One short live episode to verify sandbox wiring."""
    from curriculum.harness_runtime import HarnessRuntime, make_env
    from curriculum.harness_schema import HarnessSpec

    h = HarnessSpec()
    h.name = "ping_minestudio"
    h.runtime.backend = "minestudio"
    h.runtime.vla_mode = vla_mode
    h.runtime.max_steps = int(max_steps)
    h.probe.enabled = False
    h.recover.enabled = True
    h.recover.max_retries = 1
    h.verify.enabled = False
    print("[ping] creating MineStudio env…", flush=True)
    env = make_env(h, seed=0)
    try:
        print("[ping] reset…", flush=True)
        try:
            obs = env.reset()
        except Exception as e:
            out = {
                "ok": False,
                "stage": "reset",
                "error": f"{type(e).__name__}: {e}",
                "hint": (
                    "Gateway auth OK if you saw sandbox started; "
                    "create_env failures are usually remote image/deps "
                    "(e.g. minerl) or missing task_config — not harness wiring."
                ),
            }
            print(json.dumps(out, indent=2), flush=True)
            raise
        assert obs.get("image") is not None, "reset returned no image"
        print("[ping] step forward…", flush=True)
        obs2, reward, done, info = env.step_action_name("forward")
        out = {
            "ok": True,
            "reward": reward,
            "done": done,
            "info": {k: info[k] for k in info if k != "raw"},
            "has_image": obs2.get("image") is not None,
        }
        print(json.dumps(out, indent=2, default=str), flush=True)
        rt = HarnessRuntime(h, seed=0)
        ep = rt.run_episode(env=env, episode_seed=0)
        out["episode"] = ep.to_dict()
        print("[ping] episode:", json.dumps(ep.to_dict(), indent=2), flush=True)
        return out
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ping-minestudio", action="store_true",
                    help="Live sandbox connectivity check (1 short episode)")
    ap.add_argument("--ping-llm", action="store_true",
                    help="Ping Gemini Flash via LLM API LLM API (LLM_API_APP_ID)")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default=None, choices=["stub", "minestudio"])
    ap.add_argument("--vla-mode", default=None,
                    choices=["stub", "hf", "gemini", "gemini-flash"])
    ap.add_argument("--max-steps", type=int, default=2)
    args = ap.parse_args()

    if args.ping_gemini:
        from curriculum.gemini_vla import ping_gemini
        out = ping_gemini()
        print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
        print("\n[ping_gemini] OK", flush=True)
        return

    if args.ping_minestudio:
        ping_minestudio(max_steps=int(args.max_steps), vla_mode=str(args.vla_mode or "stub"))
        print("\n[ping_minestudio] OK", flush=True)
        return

    if args.smoke:
        logdir = args.logdir or "curriculum/outputs/smoke_auto_harness"
        if os.path.isdir(logdir):
            shutil.rmtree(logdir, ignore_errors=True)
        h = degraded_seed()
        if args.backend:
            h.runtime.backend = args.backend
        if args.vla_mode:
            h.runtime.vla_mode = args.vla_mode
        searcher = AutoHarnessSearch(
            logdir=logdir,
            seed_harness=h,
            n_rounds=int(args.rounds or 8),
            n_eval_episodes=5 if h.runtime.backend == "stub" else 2,
            seed=int(args.seed),
        )
        summary = searcher.run()
        assert "best_score" in summary
        print("\n[smoke_auto_harness] OK — training-free harness search finished.")
        print(json.dumps(summary, indent=2))
        return

    if not args.config:
        ap.error("--config, --smoke, or --ping-minestudio required")
    summary = run_from_config(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
