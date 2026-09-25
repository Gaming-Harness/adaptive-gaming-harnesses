"""Training-free contextual bandit for active probing.

Only a JSON table of probe outcomes is updated; VLA weights stay frozen.
Context estimates adapt locally while global arm estimates transfer to unseen
task/trigger contexts.
"""
from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional


def _empty_stat() -> Dict[str, float]:
    return {"pulls": 0, "reward_sum": 0.0, "immediate_sum": 0.0, "delayed_sum": 0.0}


class ContextualProbeBandit:
    """UCB bandit with a global, cross-context empirical prior."""

    def __init__(self, *, ucb_c: float = 1.2, transfer_weight: float = 0.25):
        self.ucb_c = float(ucb_c)
        self.transfer_weight = max(0.0, float(transfer_weight))
        self.contexts: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.global_arms: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _stat(table: Dict[str, Dict[str, float]], arm: str) -> Dict[str, float]:
        if arm not in table:
            table[arm] = _empty_stat()
        return table[arm]

    def select(self, context: str, arms: Iterable[str], rng: Any) -> str:
        choices = list(dict.fromkeys(str(a) for a in arms if str(a)))
        if not choices:
            raise ValueError("probe action pool is empty")
        table = self.contexts.setdefault(str(context), {})
        unseen = [a for a in choices if int(self.global_arms.get(a, {}).get("pulls", 0)) == 0]
        if unseen:
            return rng.choice(unseen)

        total = 1 + sum(int(table.get(a, {}).get("pulls", 0)) for a in choices)
        scored = []
        for arm in choices:
            local = table.get(arm, _empty_stat())
            glob = self.global_arms.get(arm, _empty_stat())
            n_local = float(local["pulls"])
            n_global = max(1.0, float(glob["pulls"]))
            global_mean = float(glob["reward_sum"]) / n_global
            prior_n = self.transfer_weight
            mean = (float(local["reward_sum"]) + prior_n * global_mean) / max(1e-9, n_local + prior_n)
            bonus = self.ucb_c * math.sqrt(math.log(float(total) + 1.0) / (n_local + 1.0))
            scored.append((mean + bonus, rng.random(), arm))
        return max(scored)[2]

    def update(self, context: str, arm: str, reward: float, *, kind: str = "immediate") -> None:
        context, arm, value = str(context), str(arm), float(reward)
        local = self._stat(self.contexts.setdefault(context, {}), arm)
        glob = self._stat(self.global_arms, arm)
        if kind == "immediate":
            local["pulls"] += 1
            glob["pulls"] += 1
            local["immediate_sum"] += value
            glob["immediate_sum"] += value
        elif kind != "delayed":
            raise ValueError(f"unknown reward kind: {kind}")
        local["reward_sum"] += value
        glob["reward_sum"] += value
        if kind == "delayed":
            local["delayed_sum"] += value
            glob["delayed_sum"] += value

    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "ucb_c": self.ucb_c, "transfer_weight": self.transfer_weight,
                "contexts": self.contexts, "global_arms": self.global_arms}

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ContextualProbeBandit":
        d = dict(data or {})
        obj = cls(ucb_c=float(d.get("ucb_c", 1.2)),
                  transfer_weight=float(d.get("transfer_weight", 0.25)))
        obj.contexts = dict(d.get("contexts") or {})
        obj.global_arms = dict(d.get("global_arms") or {})
        return obj

    @classmethod
    def load(cls, path: str, **defaults: float) -> "ContextualProbeBandit":
        if not path or not os.path.isfile(path):
            return cls(**defaults)
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        if not path:
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".probe-bandit-", suffix=".json", dir=parent)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def merge_delta_save(
        self, path: str, baseline: Mapping[str, Any]
    ) -> "ContextualProbeBandit":
        """Atomically add this rollout delta to a shared cross-process prior."""
        if not path:
            return self
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + ".lock", "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                latest = ContextualProbeBandit.load(
                    path, ucb_c=self.ucb_c, transfer_weight=self.transfer_weight
                )
                before = ContextualProbeBandit.from_dict(baseline)
                for context, arms in self.contexts.items():
                    old_arms = before.contexts.get(context, {})
                    dst_arms = latest.contexts.setdefault(context, {})
                    for arm, stat in arms.items():
                        dst = self._stat(dst_arms, arm)
                        old = old_arms.get(arm, _empty_stat())
                        for key in _empty_stat():
                            dst[key] += float(stat.get(key, 0.0)) - float(old.get(key, 0.0))
                for arm, stat in self.global_arms.items():
                    dst = self._stat(latest.global_arms, arm)
                    old = before.global_arms.get(arm, _empty_stat())
                    for key in _empty_stat():
                        dst[key] += float(stat.get(key, 0.0)) - float(old.get(key, 0.0))
                latest.save(path)
                return latest
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def summary(self) -> Dict[str, Any]:
        means = {}
        for arm, stat in self.global_arms.items():
            n = int(stat.get("pulls", 0))
            means[arm] = {"pulls": n, "mean_reward": float(stat.get("reward_sum", 0.0)) / max(1, n),
                          "immediate_sum": float(stat.get("immediate_sum", 0.0)),
                          "delayed_sum": float(stat.get("delayed_sum", 0.0))}
        return {"n_contexts": len(self.contexts), "global_arms": means}
