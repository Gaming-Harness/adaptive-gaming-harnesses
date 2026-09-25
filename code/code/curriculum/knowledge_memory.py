"""Verified world-knowledge memory — Key–Value Knowledge Slots (not raw traj / not NL).

Formal slot (see curriculum.knowledge_consolidation)::

    m_i = (C_i, A_i, E_i, conf_i)
      C = pool(z_t)                 # condition  (Key)
      A = pool([keyboard; mouse])   # action     (Key)
      E = pool(z_{t+1}) - C         # Δz residual (Value)
      conf = Probe(τ)               # written only if probes pass

``KnowledgeItem.state_embed / action_embed / effect_embed`` are exactly (C, A, E).
Text rules can be layered later for VLA prompting; the differentiable substrate
is this latent KV memory.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def pool_latent(z: torch.Tensor) -> torch.Tensor:
    """z: [..., C, f, H, W] or [C, f, H, W] → [C]."""
    if z.ndim == 5:
        return z.float().mean(dim=(2, 3, 4))
    if z.ndim == 4:
        return z.float().mean(dim=(1, 2, 3))
    raise ValueError(f"unexpected latent ndim={z.ndim}")


def action_embed(keyboard: torch.Tensor, mouse: torch.Tensor) -> torch.Tensor:
    """Mean-pool action window → [kb_dim + mouse_dim]."""
    kb = keyboard.float()
    ms = mouse.float()
    if kb.ndim == 2:
        kb = kb.mean(0)
    if ms.ndim == 2:
        ms = ms.mean(0)
    return torch.cat([kb.reshape(-1), ms.reshape(-1)], dim=0)


@dataclass
class KnowledgeItem:
    """One Knowledge Slot m_i = (C, A, E, conf). Aliases: C=state, A=action, E=effect."""

    item_id: str
    state_embed: List[float]      # C_i
    action_embed: List[float]     # A_i
    effect_embed: List[float]     # E_i = Δz
    confidence: float
    probe_scores: Dict[str, float] = field(default_factory=dict)
    verify_count: int = 1
    source: str = "probe"
    meta: Dict[str, Any] = field(default_factory=dict)
    # Auto / claw status
    status: str = "active"  # active | mastered | revoked
    fail_streak: int = 0
    last_verified_step: int = 0
    reverify_count: int = 0

    # Formal aliases for RP / Method text
    @property
    def C(self) -> List[float]:
        return self.state_embed

    @property
    def A(self) -> List[float]:
        return self.action_embed

    @property
    def E(self) -> List[float]:
        return self.effect_embed

    def state_t(self) -> torch.Tensor:
        return torch.tensor(self.state_embed, dtype=torch.float32)

    def action_t(self) -> torch.Tensor:
        return torch.tensor(self.action_embed, dtype=torch.float32)

    def effect_t(self) -> torch.Tensor:
        return torch.tensor(self.effect_embed, dtype=torch.float32)


class WorldKnowledgeMemory:
    """Growing bank of verified interaction knowledge."""

    def __init__(
        self,
        merge_thresh: float = 0.85,
        max_items: int = 5000,
        min_confidence: float = 0.55,
    ):
        self.merge_thresh = float(merge_thresh)
        self.max_items = int(max_items)
        self.min_confidence = float(min_confidence)
        self.items: List[KnowledgeItem] = []
        self.stats: Dict[str, float] = {
            "proposed": 0,
            "accepted": 0,
            "merged": 0,
            "rejected": 0,
            "mastered": 0,
            "revoked": 0,
            "reverified": 0,
        }

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())

    def nearest(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        topk: int = 1,
    ) -> List[Tuple[KnowledgeItem, float]]:
        if not self.items:
            return []
        scores = []
        for it in self.items:
            s = 0.5 * (
                self._cos(state, it.state_t()) + self._cos(action, it.action_t())
            )
            scores.append((it, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

    def novelty(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> float:
        """1 - max similarity; high = unseen pattern."""
        nn = self.nearest(state, action, topk=1)
        if not nn:
            return 1.0
        # Revoked items should not suppress novelty
        it, sim = nn[0]
        if it.status == "revoked":
            return 1.0
        return max(0.0, 1.0 - sim)

    def is_mastered_near(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        sim_thresh: float = 0.78,
    ) -> bool:
        for it, sim in self.nearest(state, action, topk=3):
            if it.status == "mastered" and sim >= sim_thresh:
                return True
        return False

    def coverage_report(
        self,
        action_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Knowledge boundary: what the WM claims to know vs not.

        This is the *self-awareness* metric — not MSE. Coverage is over action
        families (and optionally state neighborhoods), with revoke counting as
        known-but-stale rather than mastered.
        """
        if action_names is None:
            from curriculum.sandbox_experience import ACTION_NAMES
            action_names = list(ACTION_NAMES)
        per: Dict[str, Dict[str, float]] = {}
        n_mastered_fam = 0
        n_unknown_fam = 0
        n_revoked_fam = 0
        for name in action_names:
            tagged = [
                it for it in self.items
                if str((it.meta or {}).get("action", "")) == name
            ]
            n_m = sum(1 for it in tagged if it.status == "mastered")
            n_a = sum(1 for it in tagged if it.status == "active")
            n_r = sum(1 for it in tagged if it.status == "revoked")
            unknown = 1.0 if (n_m + n_a) == 0 else 0.0
            if unknown:
                n_unknown_fam += 1
            if n_m > 0:
                n_mastered_fam += 1
            if n_r > 0 and n_m == 0:
                n_revoked_fam += 1
            per[name] = {
                "n": float(len(tagged)),
                "mastered": float(n_m),
                "active": float(n_a),
                "revoked": float(n_r),
                "unknown": float(unknown),
            }
        n_fam = max(1, len(action_names))
        n_items = max(1, len(self.items))
        return {
            "n_items": float(len(self.items)),
            "n_mastered_items": float(sum(1 for it in self.items if it.status == "mastered")),
            "n_active_items": float(sum(1 for it in self.items if it.status == "active")),
            "n_revoked_items": float(sum(1 for it in self.items if it.status == "revoked")),
            "family_coverage": float(n_mastered_fam / n_fam),
            "family_unknown": float(n_unknown_fam / n_fam),
            "family_stale": float(n_revoked_fam / n_fam),
            "item_revoke_rate": float(sum(1 for it in self.items if it.status == "revoked") / n_items) if self.items else 0.0,
            "per_action": per,
        }

    def abstain_score(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        mastered_sim: float = 0.78,
    ) -> float:
        """P(I don't know) ∈ [0,1]: novelty, revoke neighborhood, or no slot.

        High abstain → OOD / stale / never probed. Used as safety signal, not MSE.
        """
        nn = self.nearest(state, action, topk=1)
        if not nn:
            return 1.0
        it, sim = nn[0]
        if it.status == "revoked":
            return 1.0
        if it.status == "mastered" and sim >= mastered_sim:
            return 0.0
        # active but weak match or low conf → partial abstain
        return float(max(0.0, min(1.0, (1.0 - sim) * (1.0 - float(it.confidence)))))

    def action_mastered_fraction(self, action_name: str) -> float:
        """Fraction of active/mastered items tagged with this action that are mastered."""
        if not action_name:
            return 0.0
        tagged = [
            it for it in self.items
            if str((it.meta or {}).get("action", "")) == action_name
            and it.status != "revoked"
        ]
        if not tagged:
            return 0.0
        mastered = sum(1 for it in tagged if it.status == "mastered")
        return float(mastered / len(tagged))

    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        for it in self.items:
            if it.item_id == item_id:
                return it
        return None

    def promote_mastered(
        self,
        min_verify: int = 2,
        min_conf: float = 0.55,
    ) -> int:
        n = 0
        for it in self.items:
            if it.status != "active":
                continue
            if it.verify_count >= min_verify and it.confidence >= min_conf and it.fail_streak == 0:
                it.status = "mastered"
                n += 1
                self.stats["mastered"] = float(self.stats.get("mastered", 0) + 1)
        return n

    def mark_mastered(self, item_id: str, step: int = 0) -> Optional[KnowledgeItem]:
        it = self.get_item(item_id)
        if it is None:
            return None
        it.status = "mastered"
        it.fail_streak = 0
        it.last_verified_step = int(step)
        self.stats["mastered"] = float(self.stats.get("mastered", 0) + 1)
        return it

    def revoke_item(
        self,
        item_id: str,
        probe_scores: Optional[Dict[str, float]] = None,
        note: str = "",
    ) -> Optional[KnowledgeItem]:
        it = self.get_item(item_id)
        if it is None:
            return None
        it.status = "revoked"
        it.fail_streak += 1
        if probe_scores:
            it.meta["revoke_scores"] = {k: float(v) for k, v in probe_scores.items()}
        if note:
            it.meta["revoke_note"] = note
        self.stats["revoked"] = float(self.stats.get("revoked", 0) + 1)
        return it

    def confirm_reverify(
        self,
        item_id: str,
        confidence: float,
        probe_scores: Optional[Dict[str, float]] = None,
        step: int = 0,
    ) -> Optional[KnowledgeItem]:
        it = self.get_item(item_id)
        if it is None:
            return None
        it.reverify_count += 1
        it.verify_count += 1
        it.confidence = max(it.confidence, float(confidence))
        it.fail_streak = 0
        it.last_verified_step = int(step)
        if probe_scores:
            it.probe_scores = {
                k: max(it.probe_scores.get(k, 0.0), float(v))
                for k, v in probe_scores.items()
            }
        if it.status == "revoked":
            it.status = "active"
        self.stats["reverified"] = float(self.stats.get("reverified", 0) + 1)
        return it

    def items_for_reverify(self, k: int = 2) -> List[KnowledgeItem]:
        """Prefer mastered / high-conf items that have not been checked recently."""
        cands = [
            it for it in self.items
            if it.status in ("mastered", "active") and it.confidence >= self.min_confidence
        ]
        cands.sort(key=lambda x: (x.last_verified_step, -x.confidence, -x.verify_count))
        return cands[: max(0, k)]

    def propose(
        self,
        latent_t: torch.Tensor,
        keyboard: torch.Tensor,
        mouse: torch.Tensor,
        latent_tp1: torch.Tensor,
        probe_scores: Dict[str, float],
        confidence: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[KnowledgeItem]:
        """Consolidate only if confidence / probes pass."""
        self.stats["proposed"] += 1
        if confidence < self.min_confidence:
            self.stats["rejected"] += 1
            return None
        # Require causal + counterfactual gates when present
        causal = probe_scores.get("action_causality", 1.0)
        cf = probe_scores.get("counterfactual", 1.0)
        transfer = probe_scores.get("state_transfer", 1.0)
        horizon = probe_scores.get("long_horizon", 1.0)
        # Soft gate: allow strong transfer/horizon to compensate early causal weakness
        gate = 0.35 * causal + 0.35 * cf + 0.15 * transfer + 0.15 * horizon
        if gate < self.min_confidence and (causal < self.min_confidence or cf < self.min_confidence):
            self.stats["rejected"] += 1
            return None

        s_emb = pool_latent(latent_t).detach().cpu().reshape(-1)
        a_emb = action_embed(keyboard, mouse).detach().cpu().reshape(-1)
        e_emb = (pool_latent(latent_tp1) - pool_latent(latent_t)).detach().cpu().reshape(-1)

        nn = self.nearest(s_emb, a_emb, topk=1)
        if nn and nn[0][1] >= self.merge_thresh and nn[0][0].status != "revoked":
            it = nn[0][0]
            # EMA-merge embeddings
            alpha = 1.0 / (it.verify_count + 1)
            it.state_embed = (
                (1 - alpha) * it.state_t() + alpha * s_emb
            ).tolist()
            it.action_embed = (
                (1 - alpha) * it.action_t() + alpha * a_emb
            ).tolist()
            it.effect_embed = (
                (1 - alpha) * it.effect_t() + alpha * e_emb
            ).tolist()
            it.confidence = max(it.confidence, float(confidence))
            it.verify_count += 1
            it.fail_streak = 0
            it.probe_scores = {
                k: max(it.probe_scores.get(k, 0.0), float(v))
                for k, v in probe_scores.items()
            }
            if meta:
                it.meta.update(meta)
                if "step" in meta:
                    it.last_verified_step = int(meta["step"])
            self.stats["merged"] += 1
            return it

        item = KnowledgeItem(
            item_id=str(uuid.uuid4())[:8],
            state_embed=s_emb.tolist(),
            action_embed=a_emb.tolist(),
            effect_embed=e_emb.tolist(),
            confidence=float(confidence),
            probe_scores={k: float(v) for k, v in probe_scores.items()},
            verify_count=1,
            meta=dict(meta or {}),
            status="active",
            last_verified_step=int((meta or {}).get("step", 0)),
        )
        self.items.append(item)
        if len(self.items) > self.max_items:
            # Drop lowest-confidence oldest-ish
            self.items.sort(key=lambda x: (x.confidence, x.verify_count))
            self.items = self.items[-self.max_items :]
        self.stats["accepted"] += 1
        return item

    def effect_prior(
        self,
        latent_t: torch.Tensor,
        keyboard: torch.Tensor,
        mouse: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Retrieve nearest verified effect as soft prior [C]."""
        s = pool_latent(latent_t).detach().cpu().reshape(-1)
        a = action_embed(keyboard, mouse).detach().cpu().reshape(-1)
        for it, sim in self.nearest(s, a, topk=5):
            if it.status == "revoked":
                continue
            if sim < self.merge_thresh * 0.9:
                break
            return it.effect_t()
        return None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "merge_thresh": self.merge_thresh,
            "max_items": self.max_items,
            "min_confidence": self.min_confidence,
            "stats": self.stats,
            "items": [asdict(it) for it in self.items],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "WorldKnowledgeMemory":
        with open(path) as f:
            payload = json.load(f)
        mem = cls(
            merge_thresh=payload.get("merge_thresh", 0.85),
            max_items=payload.get("max_items", 5000),
            min_confidence=payload.get("min_confidence", 0.55),
        )
        mem.stats = payload.get("stats", mem.stats)
        for d in payload.get("items", []):
            # backward compatible with older memory dumps
            known = {
                "item_id", "state_embed", "action_embed", "effect_embed",
                "confidence", "probe_scores", "verify_count", "source", "meta",
                "status", "fail_streak", "last_verified_step", "reverify_count",
            }
            clean = {k: v for k, v in d.items() if k in known}
            mem.items.append(KnowledgeItem(**clean))
        return mem
