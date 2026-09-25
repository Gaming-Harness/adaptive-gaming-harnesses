"""Capability hypotheses for probe-gated Auto Research.

A hypothesis is an explicit claim the experimenter will test — not a raw
trajectory. Decisions Learn / Skip / Re-verify are attached after scoring.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from curriculum.knowledge_memory import action_embed, pool_latent
from curriculum.replay_buffer import Transition
from curriculum.sandbox_experience import ACTION_NAMES, preset_mg2_action


class Decision(str, Enum):
    LEARN = "learn"
    SKIP = "skip"
    REVERIFY = "reverify"


@dataclass
class Hypothesis:
    """Claim: under condition S, action A (vs B) induces effect E."""

    hyp_id: str
    kind: str  # causal_contrast | novelty | transfer | horizon | mastered_check
    claim: str
    action_a: str
    action_b: str = "noop"
    latent_t: Optional[torch.Tensor] = None
    keyboard: Optional[torch.Tensor] = None
    mouse: Optional[torch.Tensor] = None
    latent_tp1: Optional[torch.Tensor] = None
    source: str = "propose"  # failure | novelty | under_explore | reverify | transfer
    memory_item_id: Optional[str] = None
    # scoring
    U: float = 0.0  # uncertainty / ignorance
    N: float = 0.0  # novelty
    V: float = 0.0  # value if learned (transfer / long-horizon utility)
    cost: float = 1.0
    worth_learn: float = 0.0
    decision: Decision = Decision.LEARN
    probe_focus: List[str] = field(default_factory=lambda: [
        "action_causality", "counterfactual", "state_transfer", "long_horizon",
    ])
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        d.pop("latent_t", None)
        d.pop("keyboard", None)
        d.pop("mouse", None)
        d.pop("latent_tp1", None)
        return d

    def to_transition(self) -> Optional[Transition]:
        if self.latent_t is None or self.latent_tp1 is None:
            return None
        kb = self.keyboard
        ms = self.mouse
        if kb is None or ms is None:
            kb1, ms1 = preset_mg2_action(self.action_a)
            kb, ms = kb1.unsqueeze(0), ms1.unsqueeze(0)
        return Transition(
            latent_t=self.latent_t,
            keyboard=kb,
            mouse=ms,
            latent_tp1=self.latent_tp1,
            reward=self.worth_learn,
            source="policy",
            meta={
                "hyp_id": self.hyp_id,
                "action": self.action_a,
                "action_b": self.action_b,
                "decision": self.decision.value,
                "kind": self.kind,
                "claim": self.claim,
                **self.meta,
            },
        )


class HypothesisProposer:
    """Generate explicit hypotheses from failures, novelty, and mastered checks."""

    def __init__(
        self,
        memory,
        failure_queue: Optional[List[Dict[str, Any]]] = None,
        max_failure_queue: int = 256,
    ):
        self.memory = memory
        self.failure_queue: List[Dict[str, Any]] = list(failure_queue or [])
        self.max_failure_queue = int(max_failure_queue)
        self._action_counts = {n: 0 for n in ACTION_NAMES}

    def note_failure(
        self,
        action: str,
        probe_scores: Dict[str, float],
        confidence: float,
        latent_t: Optional[torch.Tensor] = None,
        keyboard: Optional[torch.Tensor] = None,
        mouse: Optional[torch.Tensor] = None,
        latent_tp1: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.failure_queue.append({
            "action": action or "",
            "probe_scores": dict(probe_scores or {}),
            "confidence": float(confidence),
            "latent_t": latent_t,
            "keyboard": keyboard,
            "mouse": mouse,
            "latent_tp1": latent_tp1,
            "meta": dict(meta or {}),
        })
        if len(self.failure_queue) > self.max_failure_queue:
            self.failure_queue = self.failure_queue[-self.max_failure_queue :]

    def _contrast_action(self, action: str) -> str:
        opposites = {
            "forward": "back",
            "back": "forward",
            "left": "right",
            "right": "left",
            "turn_left": "turn_right",
            "turn_right": "turn_left",
            "look_up": "look_down",
            "look_down": "look_up",
            "forward_left": "forward_right",
            "forward_right": "forward_left",
            "noop": "forward",
        }
        return opposites.get(action, "noop")

    def propose(
        self,
        transitions: Sequence[Transition],
        topk: int = 12,
        reverify_k: int = 2,
    ) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []

        # 1) Failure-driven causal contrast hypotheses
        for fail in list(reversed(self.failure_queue))[: max(1, topk // 2)]:
            a = fail.get("action") or "forward"
            b = self._contrast_action(a)
            scores = fail.get("probe_scores") or {}
            weak = sorted(scores.items(), key=lambda kv: kv[1])[:2]
            focus = [k for k, _ in weak] or ["action_causality", "counterfactual"]
            hyps.append(Hypothesis(
                hyp_id=str(uuid.uuid4())[:8],
                kind="causal_contrast",
                claim=(
                    f"Under current state, action '{a}' should produce a different "
                    f"effect than '{b}' (repairing weak probes {focus})."
                ),
                action_a=a,
                action_b=b,
                latent_t=fail.get("latent_t"),
                keyboard=fail.get("keyboard"),
                mouse=fail.get("mouse"),
                latent_tp1=fail.get("latent_tp1"),
                source="failure",
                probe_focus=focus,
                meta={"prior_conf": fail.get("confidence", 0.0)},
            ))

        # 2) Novelty / under-explored from transitions
        # Offline .pt often lack action tags → synthesize diverse action hypotheses
        for t in transitions:
            name = str((t.meta or {}).get("action", "")).strip()
            if not name:
                # assign least-tried scripted action for exploration
                name = min(ACTION_NAMES, key=lambda a: (self._action_counts.get(a, 0), a))
                kb1, ms1 = preset_mg2_action(name)
                # overwrite local copies for embedding / sandbox plan
                t_kb, t_ms = kb1.unsqueeze(0), ms1.unsqueeze(0)
                synthesized = True
            else:
                t_kb, t_ms = t.keyboard, t.mouse
                synthesized = False

            self._action_counts[name] = self._action_counts.get(name, 0) + 1
            s = pool_latent(t.latent_t).cpu().reshape(-1)
            a_emb = action_embed(t_kb, t_ms).cpu().reshape(-1)
            nov = self.memory.novelty(s, a_emb)
            mastered = self.memory.is_mastered_near(s, a_emb)
            if mastered and nov < 0.25 and not synthesized:
                continue
            total = sum(self._action_counts.values()) + 1e-6
            under = 1.0 - (self._action_counts.get(name, 0) / total)
            hyps.append(Hypothesis(
                hyp_id=str(uuid.uuid4())[:8],
                kind="novelty" if (nov >= 0.4 or synthesized) else "under_explore",
                claim=(
                    f"Interaction '{name}' may be under-learned "
                    f"(novelty={nov:.2f}, under_explore={under:.2f}"
                    f"{', synthesized_action' if synthesized else ''})."
                ),
                action_a=name,
                action_b=self._contrast_action(name),
                latent_t=t.latent_t,
                keyboard=t_kb,
                mouse=t_ms,
                latent_tp1=t.latent_tp1,
                source="novelty" if (nov >= 0.4 or synthesized) else "under_explore",
                N=float(max(nov, 0.55 if synthesized else nov)),
                meta={"under_explore": float(under), "synthesized_action": synthesized},
            ))

        # 3) Explicit under-explored action family sweep (ensures Learn has fuel)
        ranked_actions = sorted(
            ACTION_NAMES,
            key=lambda a: (self._action_counts.get(a, 0), a),
        )
        for name in ranked_actions[: max(2, topk // 3)]:
            if name == "noop":
                continue
            if any(h.action_a == name and h.source in ("novelty", "under_explore") for h in hyps):
                continue
            hyps.append(Hypothesis(
                hyp_id=str(uuid.uuid4())[:8],
                kind="under_explore",
                claim=f"Action family '{name}' is under-explored in sandbox curriculum.",
                action_a=name,
                action_b=self._contrast_action(name),
                source="under_explore",
                N=0.7,
                meta={"family_sweep": True},
            ))

        # 4) Scheduled re-verify of mastered / high-conf memory
        for it in self.memory.items_for_reverify(k=reverify_k):
            act = str((it.meta or {}).get("action", "")) or "noop"
            hyps.append(Hypothesis(
                hyp_id=str(uuid.uuid4())[:8],
                kind="mastered_check",
                claim=(
                    f"Re-verify mastered/active knowledge '{it.item_id}' "
                    f"(action={act}, conf={it.confidence:.2f})."
                ),
                action_a=act,
                action_b=self._contrast_action(act),
                source="reverify",
                memory_item_id=it.item_id,
                decision=Decision.REVERIFY,
                probe_focus=["action_causality", "counterfactual", "state_transfer"],
                meta={"verify_count": it.verify_count, "status": it.status},
            ))

        # Deduplicate by (kind, action_a, source) keeping first
        seen = set()
        uniq: List[Hypothesis] = []
        for h in hyps:
            key = (h.kind, h.action_a, h.source, h.memory_item_id or "")
            if key in seen:
                continue
            seen.add(key)
            uniq.append(h)
        return uniq[: max(1, topk)]
