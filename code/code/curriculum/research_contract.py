"""Frozen research contract for Gaming WM Auto Research MVP.

Agent / evolution code MUST NOT mutate these fields after lock().
Covers: seeds, step budget, frozen diagnostics, data splits, pass gates.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Keys that are part of the immutable scientific contract.
FROZEN_KEYS = (
    "seed",
    "max_steps",
    "max_sandbox_episodes",
    "max_wm_updates",
    "probe_suite",
    "pass_thresh",
    "eval_n",
    "transfer_split",
    "data_root_train",
    "data_root_holdout",
    "hidden_tasks",
    "mechanism_id",
)


@dataclass
class TransferSplit:
    """One frozen transfer dimension for gaming WM (task / map / action-dist)."""
    name: str = "action_family"
    train_actions: List[str] = field(
        default_factory=lambda: ["forward", "back", "left", "right", "noop"]
    )
    holdout_actions: List[str] = field(
        default_factory=lambda: ["turn_left", "turn_right", "look_up", "look_down",
                                 "forward_left", "forward_right"]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchContract:
    """Immutable budget + evaluation contract (locked after init)."""
    seed: int = 0
    max_steps: int = 50
    max_sandbox_episodes: int = 50
    max_wm_updates: int = 50
    probe_suite: Tuple[str, ...] = (
        "action_causality",
        "counterfactual",
        "state_transfer",
        "long_horizon",
        "holdout_transfer",
    )
    pass_thresh: float = 0.40
    eval_n: int = 32
    transfer_split: TransferSplit = field(default_factory=TransferSplit)
    data_root_train: str = "data/mc_vpt_train"
    data_root_holdout: str = "data/mc_vpt_pt"
    hidden_tasks: List[str] = field(default_factory=list)
    mechanism_id: str = "causal_intervene_selective_update"
    domain: str = "gaming_world_model"
    notes: str = "Agent cannot modify evaluator / budget / seeds after lock."

    # runtime counters (mutable, but capped by frozen maxima)
    steps_used: int = 0
    sandbox_episodes_used: int = 0
    wm_updates_used: int = 0

    _locked: bool = False
    _fingerprint: str = ""

    def fingerprint(self) -> str:
        payload = {
            k: getattr(self, k)
            for k in FROZEN_KEYS
            if k != "transfer_split"
        }
        payload["transfer_split"] = self.transfer_split.to_dict()
        payload["domain"] = self.domain
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def lock(self) -> "ResearchContract":
        self._fingerprint = self.fingerprint()
        self._locked = True
        return self

    def assert_locked(self) -> None:
        if not self._locked:
            raise RuntimeError("ResearchContract must be lock()ed before runs")
        if self.fingerprint() != self._fingerprint:
            raise RuntimeError(
                "FROZEN contract mutated after lock "
                f"(was {self._fingerprint}, now {self.fingerprint()}). "
                "Agent is forbidden to change evaluator/budget/seeds."
            )

    def bump_step(self) -> None:
        self.assert_locked()
        self.steps_used += 1
        if self.steps_used > self.max_steps:
            raise RuntimeError(
                f"budget exceeded: steps {self.steps_used}/{self.max_steps}"
            )

    def bump_sandbox(self) -> None:
        self.assert_locked()
        self.sandbox_episodes_used += 1
        if self.sandbox_episodes_used > self.max_sandbox_episodes:
            raise RuntimeError(
                f"budget exceeded: sandbox episodes "
                f"{self.sandbox_episodes_used}/{self.max_sandbox_episodes}"
            )

    def bump_wm_update(self) -> None:
        self.assert_locked()
        self.wm_updates_used += 1
        if self.wm_updates_used > self.max_wm_updates:
            raise RuntimeError(
                f"budget exceeded: wm updates "
                f"{self.wm_updates_used}/{self.max_wm_updates}"
            )

    def budget_left(self) -> Dict[str, int]:
        return {
            "steps": self.max_steps - self.steps_used,
            "sandbox_episodes": self.max_sandbox_episodes - self.sandbox_episodes_used,
            "wm_updates": self.max_wm_updates - self.wm_updates_used,
        }

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "seed": self.seed,
            "max_steps": self.max_steps,
            "max_sandbox_episodes": self.max_sandbox_episodes,
            "max_wm_updates": self.max_wm_updates,
            "probe_suite": list(self.probe_suite),
            "pass_thresh": self.pass_thresh,
            "eval_n": self.eval_n,
            "transfer_split": self.transfer_split.to_dict(),
            "data_root_train": self.data_root_train,
            "data_root_holdout": self.data_root_holdout,
            "hidden_tasks": list(self.hidden_tasks),
            "mechanism_id": self.mechanism_id,
            "domain": self.domain,
            "notes": self.notes,
            "locked": self._locked,
            "fingerprint": self._fingerprint or self.fingerprint(),
            "usage": {
                "steps_used": self.steps_used,
                "sandbox_episodes_used": self.sandbox_episodes_used,
                "wm_updates_used": self.wm_updates_used,
            },
            "budget_left": self.budget_left(),
        }
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_cfg(cls, cfg) -> "ResearchContract":
        ts = cfg.get("transfer_split", None)
        if ts is None:
            split = TransferSplit()
        elif isinstance(ts, TransferSplit):
            split = ts
        else:
            split = TransferSplit(
                name=str(ts.get("name", "action_family")),
                train_actions=list(ts.get("train_actions", TransferSplit().train_actions)),
                holdout_actions=list(ts.get("holdout_actions", TransferSplit().holdout_actions)),
            )
        return cls(
            seed=int(cfg.get("seed", 0)),
            max_steps=int(cfg.get("max_steps", 50)),
            max_sandbox_episodes=int(cfg.get("max_sandbox_episodes", cfg.get("max_steps", 50))),
            max_wm_updates=int(cfg.get("max_wm_updates", cfg.get("max_steps", 50))),
            probe_suite=tuple(
                cfg.get(
                    "probe_suite",
                    (
                        "action_causality",
                        "counterfactual",
                        "state_transfer",
                        "long_horizon",
                        "holdout_transfer",
                    ),
                )
            ),
            pass_thresh=float(cfg.get("pass_thresh", 0.40)),
            eval_n=int(cfg.get("eval_n", 32)),
            transfer_split=split,
            data_root_train=str(cfg.get("data_root_train", cfg.get("data_root", "data/mc_vpt_train"))),
            data_root_holdout=str(cfg.get("data_root_holdout", "data/mc_vpt_pt")),
            hidden_tasks=list(cfg.get("hidden_tasks", [])),
            mechanism_id=str(cfg.get("mechanism_id", "causal_intervene_selective_update")),
            domain=str(cfg.get("domain", "gaming_world_model")),
        )


def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
