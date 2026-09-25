"""Probe-gated Auto scheduler: Learn / Skip / Re-verify.

WorthLearn(H) = U(H) * N(H) * V(H) - λ * Cost(H)

  LEARN     — spend sandbox + selective WM update budget
  SKIP      — already known / low information gain; do not retrain
  REVERIFY  — previously trusted; check for revoke / keep mastered
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from curriculum.hypothesis import Decision, Hypothesis, HypothesisProposer
from curriculum.knowledge_memory import action_embed, pool_latent
from curriculum.replay_buffer import Transition
from curriculum.sandbox_experience import ACTION_NAMES, preset_mg2_action


@dataclass
class SchedulePlan:
    hypotheses: List[Hypothesis]
    n_learn: int = 0
    n_skip: int = 0
    n_reverify: int = 0
    action_plan: List[Dict[str, Any]] = field(default_factory=list)
    probe_focus: List[str] = field(default_factory=list)
    update_mode: str = "learn"  # learn | skip | reverify | mixed
    metrics: Dict[str, float] = field(default_factory=dict)


class AutoScheduler:
    """Central experimenter: decide what is worth learning."""

    def __init__(
        self,
        memory,
        student: Optional[torch.nn.Module] = None,
        teacher: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        lambda_cost: float = 0.15,
        skip_thresh: float = 0.18,
        learn_thresh: float = 0.28,
        mastered_sim: float = 0.78,
        mastered_min_verify: int = 2,
        mastered_min_conf: float = 0.55,
        reverify_every: int = 8,
        cost_sandbox: float = 1.0,
    ):
        self.memory = memory
        self.student = student
        self.teacher = teacher
        self.device = device or torch.device("cpu")
        self.lambda_cost = float(lambda_cost)
        self.skip_thresh = float(skip_thresh)
        self.learn_thresh = float(learn_thresh)
        self.mastered_sim = float(mastered_sim)
        self.mastered_min_verify = int(mastered_min_verify)
        self.mastered_min_conf = float(mastered_min_conf)
        self.reverify_every = int(reverify_every)
        self.cost_sandbox = float(cost_sandbox)
        self.proposer = HypothesisProposer(memory)
        self.step = 0
        self.stats = {
            "learn": 0.0,
            "skip": 0.0,
            "reverify": 0.0,
            "sandbox_turns_planned": 0.0,
            "wm_updates_gated_off": 0.0,
        }

    def note_probe_failure(self, **kwargs) -> None:
        self.proposer.note_failure(**kwargs)

    def _uncertainty(
        self,
        z: Optional[torch.Tensor],
        kb: Optional[torch.Tensor],
        ms: Optional[torch.Tensor],
        z_next: Optional[torch.Tensor],
        prior_conf: float = 0.0,
    ) -> float:
        if z is None or kb is None or ms is None or z_next is None or self.student is None:
            # failure-driven: low prior confidence → high U
            return float(max(0.0, min(1.0, 1.0 - prior_conf)))
        with torch.no_grad():
            zz = z.unsqueeze(0).to(self.device).float()
            k = kb.mean(0) if kb.ndim == 2 else kb
            m = ms.mean(0) if ms.ndim == 2 else ms
            k = k.unsqueeze(0).to(self.device).float()
            m = m.unsqueeze(0).to(self.device).float()
            pred = self.student(zz, k, m)
            err = float(torch.nn.functional.mse_loss(
                pred, z_next.unsqueeze(0).to(self.device).float()
            ).item())
            disagree = 0.0
            if self.teacher is not None:
                pred_t = self.teacher(zz, k, m)
                disagree = float(torch.nn.functional.mse_loss(pred, pred_t).item())
            return float(min(1.0, 0.55 * err + 0.45 * disagree + 0.25 * (1.0 - prior_conf)))

    def _novelty(self, h: Hypothesis) -> float:
        if h.N > 0:
            return float(h.N)
        if h.latent_t is None or h.keyboard is None or h.mouse is None:
            # unknown state → treat as moderately novel unless mastered action family
            if self.memory.action_mastered_fraction(h.action_a) >= 0.8:
                return 0.15
            return 0.55
        s = pool_latent(h.latent_t).cpu().reshape(-1)
        a = action_embed(h.keyboard, h.mouse).cpu().reshape(-1)
        return float(self.memory.novelty(s, a))

    def _value(self, h: Hypothesis) -> float:
        """If learned, how useful? Prefer transfer/horizon-sensitive kinds + rare actions."""
        base = {
            "causal_contrast": 0.85,
            "novelty": 0.70,
            "under_explore": 0.65,
            "transfer": 0.90,
            "horizon": 0.80,
            "mastered_check": 0.40,
        }.get(h.kind, 0.55)
        # boost actions that appear in holdout family (camera / compound)
        holdoutish = {
            "turn_left", "turn_right", "look_up", "look_down",
            "forward_left", "forward_right",
        }
        if h.action_a in holdoutish:
            base = min(1.0, base + 0.15)
        # downweight noop
        if h.action_a == "noop":
            base *= 0.5
        return float(base)

    def score(self, h: Hypothesis) -> Hypothesis:
        prior = float((h.meta or {}).get("prior_conf", 0.0))
        h.U = self._uncertainty(h.latent_t, h.keyboard, h.mouse, h.latent_tp1, prior)
        h.N = self._novelty(h)
        h.V = self._value(h)
        h.cost = self.cost_sandbox * (1.0 + 0.25 * len(h.probe_focus))
        h.worth_learn = float(h.U * h.N * h.V - self.lambda_cost * h.cost)
        return h

    def decide(self, h: Hypothesis) -> Hypothesis:
        self.score(h)
        # Forced re-verify path
        if h.source == "reverify" or h.kind == "mastered_check":
            h.decision = Decision.REVERIFY
            return h

        # Mastered neighborhood → Skip unless worth is very high (surprise)
        if h.latent_t is not None and h.keyboard is not None and h.mouse is not None:
            s = pool_latent(h.latent_t).cpu().reshape(-1)
            a = action_embed(h.keyboard, h.mouse).cpu().reshape(-1)
            if self.memory.is_mastered_near(s, a, sim_thresh=self.mastered_sim):
                if h.worth_learn < self.learn_thresh * 1.5:
                    h.decision = Decision.SKIP
                    h.meta["skip_reason"] = "mastered_near"
                    return h

        # Action family mostly mastered + low novelty
        if (
            self.memory.action_mastered_fraction(h.action_a) >= 0.75
            and h.N < 0.25
            and h.U < 0.35
        ):
            h.decision = Decision.SKIP
            h.meta["skip_reason"] = "action_family_mastered"
            return h

        # Failure-driven always Learn
        if h.source == "failure":
            h.decision = Decision.LEARN
            return h

        if h.worth_learn < self.skip_thresh:
            h.decision = Decision.SKIP
            h.meta["skip_reason"] = "low_worth_learn"
            return h

        if h.worth_learn >= self.learn_thresh or h.source in ("novelty", "under_explore"):
            # cold-start / exploration: novelty and family sweep should Learn
            if h.source in ("novelty", "under_explore") and h.worth_learn >= self.skip_thresh * 0.5:
                h.decision = Decision.LEARN
                return h
            if h.worth_learn >= self.learn_thresh:
                h.decision = Decision.LEARN
                return h

        h.decision = Decision.SKIP
        h.meta["skip_reason"] = "gray_zone"
        return h

    def build_plan(
        self,
        transitions: Sequence[Transition],
        topk: int = 12,
        sandbox_turns: int = 8,
        step: int = 0,
    ) -> SchedulePlan:
        self.step = int(step)
        self.proposer.memory = self.memory
        reverify_k = 2 if (self.step > 0 and self.step % self.reverify_every == 0) else (
            1 if len(self.memory) > 0 else 0
        )
        raw = self.proposer.propose(transitions, topk=topk, reverify_k=reverify_k)
        decided = [self.decide(h) for h in raw]

        # Cold start: if memory empty and nothing to Learn, force top novelty → Learn
        if len(self.memory) == 0 and not any(h.decision == Decision.LEARN for h in decided):
            forced = sorted(
                [h for h in decided if h.decision != Decision.REVERIFY],
                key=lambda h: h.worth_learn,
                reverse=True,
            )[: max(2, sandbox_turns // 2)]
            for h in forced:
                h.decision = Decision.LEARN
                h.meta["skip_reason"] = ""
                h.meta["forced_cold_start"] = True

        learns = [h for h in decided if h.decision == Decision.LEARN]
        skips = [h for h in decided if h.decision == Decision.SKIP]
        reverifies = [h for h in decided if h.decision == Decision.REVERIFY]

        self.stats["learn"] += len(learns)
        self.stats["skip"] += len(skips)
        self.stats["reverify"] += len(reverifies)

        # Sandbox action plan: prioritize Learn, then Reverify; never schedule Skip
        action_plan: List[Dict[str, Any]] = []
        ordered = sorted(
            learns + reverifies,
            key=lambda h: h.worth_learn,
            reverse=True,
        )
        for h in ordered:
            if len(action_plan) >= sandbox_turns:
                break
            action_plan.append({
                "action": h.action_a,
                "intervene_b": h.action_b,
                "decision": h.decision.value,
                "hyp_id": h.hyp_id,
                "kind": h.kind,
                "worth_learn": h.worth_learn,
            })
        self.stats["sandbox_turns_planned"] += len(action_plan)

        focus: List[str] = []
        for h in learns + reverifies:
            for p in h.probe_focus:
                if p not in focus:
                    focus.append(p)
        if not focus:
            focus = ["action_causality", "counterfactual"]

        if learns and reverifies:
            mode = "mixed"
        elif learns:
            mode = "learn"
        elif reverifies:
            mode = "reverify"
        else:
            mode = "skip"
            self.stats["wm_updates_gated_off"] += 1.0

        metrics = {
            "auto/n_hyp": float(len(decided)),
            "auto/n_learn": float(len(learns)),
            "auto/n_skip": float(len(skips)),
            "auto/n_reverify": float(len(reverifies)),
            "auto/mean_worth_learn": float(
                sum(h.worth_learn for h in decided) / max(1, len(decided))
            ),
            "auto/mean_U": float(sum(h.U for h in decided) / max(1, len(decided))),
            "auto/mean_N": float(sum(h.N for h in decided) / max(1, len(decided))),
            "auto/mean_V": float(sum(h.V for h in decided) / max(1, len(decided))),
            "auto/sandbox_turns": float(len(action_plan)),
            "auto/update_mode_learn": 1.0 if mode in ("learn", "mixed") else 0.0,
            "auto/update_mode_skip": 1.0 if mode == "skip" else 0.0,
        }
        return SchedulePlan(
            hypotheses=decided,
            n_learn=len(learns),
            n_skip=len(skips),
            n_reverify=len(reverifies),
            action_plan=action_plan,
            probe_focus=focus,
            update_mode=mode,
            metrics=metrics,
        )

    def promote_mastered(self) -> int:
        """Mark high-verify memory items as mastered."""
        return self.memory.promote_mastered(
            min_verify=self.mastered_min_verify,
            min_conf=self.mastered_min_conf,
        )
