"""Capability discovery → probe → experience compression.

This is the RP spine (WM-only):

  1. Discover  — automatically propose candidate interaction capabilities
  2. Probe     — intervention tests: is the capability real / useful?
  3. Compress  — keep only verified knowledge in experience memory
                (drop raw trajectories / failed hypotheses)

A *capability candidate* is a compact hypothesis:
  (condition state, action, predicted effect, discovery score)
not a full episode.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from curriculum.knowledge_memory import (
    WorldKnowledgeMemory,
    action_embed,
    pool_latent,
)
from curriculum.probes import WorldModelProber, _mean_action
from curriculum.replay_buffer import Transition
from curriculum.sandbox_experience import (
    ACTION_NAMES,
    InterventionPair,
    expand_action_window,
    preset_mg2_action,
    sample_scripted_action,
)


@dataclass
class CapabilityCandidate:
    """One auto-discovered capability hypothesis (pre-verification)."""
    cand_id: str
    latent_t: torch.Tensor
    keyboard: torch.Tensor
    mouse: torch.Tensor
    latent_tp1: torch.Tensor          # Env / observed effect
    action_name: str = ""
    discovery_score: float = 0.0      # novelty + uncertainty + under-explore
    novelty: float = 0.0
    uncertainty: float = 0.0
    source: str = "discover"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_transition(self) -> Transition:
        return Transition(
            latent_t=self.latent_t,
            keyboard=self.keyboard,
            mouse=self.mouse,
            latent_tp1=self.latent_tp1,
            reward=self.discovery_score,
            source="policy",
            meta={
                "cand_id": self.cand_id,
                "action": self.action_name,
                "discovery_score": self.discovery_score,
                "novelty": self.novelty,
                "uncertainty": self.uncertainty,
                **self.meta,
            },
        )


@dataclass
class VerifiedCapability:
    """Probe-passed capability ready for experience compression."""
    cand_id: str
    probe_scores: Dict[str, float]
    confidence: float
    compressed: bool = False
    item_id: Optional[str] = None


class CapabilityDiscoverer:
    """Auto-propose capabilities the WM has *not* yet mastered."""

    def __init__(
        self,
        memory: WorldKnowledgeMemory,
        student: torch.nn.Module,
        teacher: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        under_explore_bonus: float = 0.3,
    ):
        self.memory = memory
        self.student = student
        self.teacher = teacher
        self.device = device or torch.device("cpu")
        self.under_explore_bonus = float(under_explore_bonus)
        self._action_counts = {n: 0 for n in ACTION_NAMES}
        self.stats = {
            "discovered": 0,
            "probed": 0,
            "verified": 0,
            "compressed": 0,
            "discarded": 0,
        }

    def _uncertainty(
        self,
        z: torch.Tensor,
        kb: torch.Tensor,
        ms: torch.Tensor,
        z_next: torch.Tensor,
    ) -> float:
        """Disagreement student↔teacher + prediction error as uncertainty."""
        with torch.no_grad():
            zz = z.unsqueeze(0).to(self.device).float()
            k = kb.mean(0) if kb.ndim == 2 else kb
            m = ms.mean(0) if ms.ndim == 2 else ms
            k = k.unsqueeze(0).to(self.device).float()
            m = m.unsqueeze(0).to(self.device).float()
            pred_s = self.student(zz, k, m)
            err = float(F.mse_loss(pred_s, z_next.unsqueeze(0).to(self.device).float()).item())
            if self.teacher is None:
                return float(min(1.0, err))
            pred_t = self.teacher(zz, k, m)
            disagree = float(F.mse_loss(pred_s, pred_t).item())
            return float(min(1.0, 0.5 * err + 0.5 * disagree))

    def score_transition(self, t: Transition, action_name: str = "") -> Tuple[float, float, float]:
        name = action_name or str((t.meta or {}).get("action", ""))
        nov = self.memory.novelty(
            pool_latent(t.latent_t).cpu(),
            action_embed(t.keyboard, t.mouse).cpu(),
        )
        unc = self._uncertainty(t.latent_t, t.keyboard, t.mouse, t.latent_tp1)
        # prefer rarely tried action types
        total = sum(self._action_counts.values()) + 1e-6
        freq = self._action_counts.get(name, 0) / total if name else 0.0
        under = self.under_explore_bonus * (1.0 - freq)
        score = float(0.45 * nov + 0.40 * unc + 0.15 * under)
        return score, nov, unc

    def discover_from_transitions(
        self,
        transitions: Sequence[Transition],
        topk: int = 8,
    ) -> List[CapabilityCandidate]:
        """Rank sandbox / buffer transitions as capability candidates."""
        scored: List[CapabilityCandidate] = []
        for t in transitions:
            name = str((t.meta or {}).get("action", ""))
            score, nov, unc = self.score_transition(t, name)
            if name in self._action_counts:
                self._action_counts[name] += 1
            scored.append(CapabilityCandidate(
                cand_id=str(uuid.uuid4())[:8],
                latent_t=t.latent_t,
                keyboard=t.keyboard,
                mouse=t.mouse,
                latent_tp1=t.latent_tp1,
                action_name=name,
                discovery_score=score,
                novelty=nov,
                uncertainty=unc,
                source="discover",
                meta=dict(t.meta or {}),
            ))
        scored.sort(key=lambda c: c.discovery_score, reverse=True)
        out = scored[: max(1, topk)]
        self.stats["discovered"] += len(out)
        return out

    def propose_actions_for_state(
        self,
        latent_t: torch.Tensor,
        n: int = 3,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """Auto-pick under-explored / diverse actions to try next in sandbox."""
        # least-tried actions first, with light randomness
        ranked = sorted(ACTION_NAMES, key=lambda a: (self._action_counts[a], random_jitter(a)))
        picks = []
        for name in ranked:
            if len(picks) >= n:
                break
            if name == "noop" and len(picks) > 0:
                continue
            kb, ms = preset_mg2_action(name)
            picks.append((name, kb, ms))
        return picks

    def probe_candidates(
        self,
        prober: WorldModelProber,
        candidates: Sequence[CapabilityCandidate],
        pool: Sequence[Transition],
        interventions: Optional[Sequence[InterventionPair]] = None,
        pass_thresh: float = 0.45,
    ) -> Tuple[List[VerifiedCapability], List[CapabilityCandidate]]:
        """Probe each candidate; split into verified vs discarded."""
        verified: List[VerifiedCapability] = []
        discarded: List[CapabilityCandidate] = []
        inter = list(interventions or [])
        for c in candidates:
            self.stats["probed"] += 1
            tr = c.to_transition()
            # prefer intervention pairs that share this state/action when available
            local_inter = [
                p for p in inter
                if p.name_a == c.action_name or p.name_b == c.action_name
            ][:4]
            result = prober.probe_batch(
                [tr],
                pool if pool else [tr],
                chains=None,
                interventions=local_inter if local_inter else (inter[:4] if inter else None),
            )
            conf = result.confidence
            if conf >= pass_thresh:
                verified.append(VerifiedCapability(
                    cand_id=c.cand_id,
                    probe_scores=dict(result.scores),
                    confidence=conf,
                ))
                self.stats["verified"] += 1
            else:
                discarded.append(c)
                self.stats["discarded"] += 1
                # annotate for priority replay
                c.meta["probe_conf"] = conf
                c.meta["probe_scores"] = dict(result.scores)
        return verified, discarded

    def compress(
        self,
        candidates: Sequence[CapabilityCandidate],
        verified: Sequence[VerifiedCapability],
    ) -> List[str]:
        """Compress verified candidates into experience memory; drop the rest."""
        by_id = {c.cand_id: c for c in candidates}
        item_ids: List[str] = []
        for v in verified:
            c = by_id.get(v.cand_id)
            if c is None:
                continue
            item = self.memory.propose(
                c.latent_t,
                c.keyboard,
                c.mouse,
                c.latent_tp1,
                probe_scores=v.probe_scores,
                confidence=v.confidence,
                meta={
                    "cand_id": c.cand_id,
                    "action": c.action_name,
                    "discovery_score": c.discovery_score,
                    "novelty": c.novelty,
                    "uncertainty": c.uncertainty,
                    "compressed": True,
                    **c.meta,
                },
            )
            if item is not None:
                v.compressed = True
                v.item_id = item.item_id
                item_ids.append(item.item_id)
                self.stats["compressed"] += 1
        return item_ids


def random_jitter(name: str) -> float:
    # deterministic-ish tie-break without importing random at module score time
    return (hash(name) % 1000) / 1000.0


def discover_probe_compress(
    discoverer: CapabilityDiscoverer,
    prober: WorldModelProber,
    transitions: Sequence[Transition],
    pool: Sequence[Transition],
    interventions: Optional[Sequence[InterventionPair]] = None,
    topk: int = 8,
    pass_thresh: float = 0.45,
) -> Dict[str, Any]:
    """One full Discover → Probe → Compress cycle."""
    cands = discoverer.discover_from_transitions(transitions, topk=topk)
    verified, discarded = discoverer.probe_candidates(
        prober, cands, pool, interventions=interventions, pass_thresh=pass_thresh,
    )
    item_ids = discoverer.compress(cands, verified)
    return {
        "candidates": cands,
        "verified": verified,
        "discarded": discarded,
        "compressed_ids": item_ids,
        "stats": dict(discoverer.stats),
        "n_discovered": len(cands),
        "n_verified": len(verified),
        "n_discarded": len(discarded),
        "n_compressed": len(item_ids),
    }
