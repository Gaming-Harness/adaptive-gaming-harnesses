"""Intervention-sensitivity probes (NOT causal discovery / SEM / IV).

Offline mode uses expert .pt or already-collected τ as Env(s,a) ground truth.
Most tests roll F_φ in *latent* with a vs a' — no extra sandbox reset per step.
Optional true sandbox reset is grounding, not the train loop.

Probes
------
1. Action sensitivity — observed a should fit s* better than intervened a'
2. Counterfactual     — different actions → different futures when GT differs
3. State transfer     — similar actions → similar residual effects across states
4. Long-horizon       — multi-step latent imagination drift vs GT (uncertainty-aware)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from curriculum.knowledge_memory import action_embed, pool_latent
from curriculum.replay_buffer import Transition


EnvStepFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]  # (z, kb, ms) -> z_next  (ground truth / sandbox)


def _mean_action(kb: torch.Tensor, ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if kb.ndim == 2:
        kb = kb.mean(0)
    if ms.ndim == 2:
        ms = ms.mean(0)
    return kb.float(), ms.float()


def _action_l1(kb1: torch.Tensor, ms1: torch.Tensor, kb2: torch.Tensor, ms2: torch.Tensor) -> float:
    return float((kb1 - kb2).abs().sum() + (ms1 - ms2).abs().sum())


def _synth_action(
    kb: torch.Tensor,
    ms: torch.Tensor,
    kb_dim: int = 4,
    mouse_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard intervention: flip / resample action away from the observed one."""
    # keyboard: one-hot-ish — pick a different active key
    kb_new = torch.zeros_like(kb)
    idx = int(torch.randint(0, kb_dim, (1,)).item())
    # ensure different from argmax of original when possible
    orig = int(kb.argmax().item()) if kb.numel() == kb_dim else -1
    if idx == orig:
        idx = (idx + 1) % kb_dim
    kb_new[idx] = 1.0
    # mouse: large random look
    ms_new = torch.randn_like(ms) * 0.5
    if _action_l1(kb, ms, kb_new, ms_new) < 1e-5:
        ms_new = ms + torch.tensor([0.3, -0.3], dtype=ms.dtype)
    return kb_new.float(), ms_new.float()


def _sample_intervened_action(
    kb_pos: torch.Tensor,
    ms_pos: torch.Tensor,
    action_pool: Sequence[Transition],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefer a real alternate action from the pool; fall back to synthetic."""
    for _ in range(min(8, max(1, len(action_pool)))):
        neg = random.choice(action_pool)
        kb_n, ms_n = _mean_action(neg.keyboard, neg.mouse)
        if _action_l1(kb_pos, ms_pos, kb_n, ms_n) > 1e-5:
            return kb_n, ms_n
    return _synth_action(kb_pos, ms_pos)


@dataclass
class ProbeResult:
    scores: Dict[str, float]
    details: Dict[str, float] = field(default_factory=dict)
    per_item: List[Dict[str, float]] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        if not self.scores:
            return 0.0
        return float(sum(self.scores.values()) / len(self.scores))


class WorldModelProber:
    """Evaluate what a dynamics model has actually learned."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        horizon: int = 3,
        n_intervene: int = 3,
        transfer_pairs: int = 16,
        causal_margin: float = 1e-4,
    ):
        self.model = model
        self.device = device
        self.horizon = int(horizon)
        self.n_intervene = int(n_intervene)
        self.transfer_pairs = int(transfer_pairs)
        self.causal_margin = float(causal_margin)

    def _predict(
        self,
        z: torch.Tensor,
        kb: torch.Tensor,
        ms: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(z.float(), kb.float(), ms.float())

    @torch.no_grad()
    def action_causality(
        self,
        batch: Sequence[Transition],
        action_pool: Sequence[Transition],
    ) -> Tuple[float, List[Dict[str, float]]]:
        """Intervene on action at fixed s_t; correct a* should match Env better."""
        if not batch or not action_pool:
            return 0.0, []
        scores = []
        per = []
        for t in batch:
            z = t.latent_t.unsqueeze(0).to(self.device).float()
            z_star = t.latent_tp1.unsqueeze(0).to(self.device).float()
            kb_pos, ms_pos = _mean_action(t.keyboard, t.mouse)
            kb_pos = kb_pos.unsqueeze(0).to(self.device)
            ms_pos = ms_pos.unsqueeze(0).to(self.device)
            pred_pos = self._predict(z, kb_pos, ms_pos)
            err_pos = float(F.mse_loss(pred_pos, z_star).item())

            wins = 0
            trials = 0
            sep = 0.0
            for _ in range(self.n_intervene):
                kb_n, ms_n = _sample_intervened_action(
                    kb_pos.squeeze(0).cpu(), ms_pos.squeeze(0).cpu(), action_pool
                )
                kb_n = kb_n.unsqueeze(0).to(self.device)
                ms_n = ms_n.unsqueeze(0).to(self.device)
                pred_neg = self._predict(z, kb_n, ms_n)
                err_neg = float(F.mse_loss(pred_neg, z_star).item())
                sep += float(F.mse_loss(pred_pos, pred_neg).item())
                trials += 1
                if err_pos + self.causal_margin <= err_neg:
                    wins += 1
            if trials == 0:
                continue
            win_rate = wins / trials
            # Controllability: intervened actions must change the predicted future
            mean_sep = sep / trials
            ctrl = float(min(1.0, mean_sep / (err_pos + 1e-4)))
            score = 0.7 * win_rate + 0.3 * ctrl
            scores.append(score)
            per.append({
                "action_causality": score,
                "err_pos": err_pos,
                "win_rate": win_rate,
                "ctrl": ctrl,
            })
        if not scores:
            return 0.0, []
        return float(sum(scores) / len(scores)), per

    @torch.no_grad()
    def counterfactual(
        self,
        batch: Sequence[Transition],
        action_pool: Sequence[Transition],
    ) -> float:
        """Does F distinguish alternate futures when actions differ?"""
        if len(batch) < 2:
            return 0.0
        scores = []
        for t in batch:
            z = t.latent_t.unsqueeze(0).to(self.device).float()
            kb1, ms1 = _mean_action(t.keyboard, t.mouse)
            kb2, ms2 = _sample_intervened_action(kb1, ms1, action_pool)
            a_dist = _action_l1(kb1, ms1, kb2, ms2)
            pred1 = self._predict(
                z, kb1.unsqueeze(0).to(self.device), ms1.unsqueeze(0).to(self.device)
            )
            pred2 = self._predict(
                z, kb2.unsqueeze(0).to(self.device), ms2.unsqueeze(0).to(self.device)
            )
            s_dist = float(F.mse_loss(pred1, pred2).item())
            # Controllable dynamics: nonzero action intervention → nonzero Δŝ
            if a_dist < 1e-5:
                scores.append(0.0)
                continue
            # Score saturates as predicted futures separate under intervention
            scores.append(float(min(1.0, s_dist / (s_dist + 1e-3))))
        return float(sum(scores) / max(len(scores), 1))

    @torch.no_grad()
    def state_transfer(self, pool: Sequence[Transition]) -> float:
        """Similar actions → similar residual effects across different states."""
        if len(pool) < 4:
            return 0.0
        pairs = min(self.transfer_pairs, len(pool) * 2)
        sims = []
        for _ in range(pairs):
            t1, t2 = random.sample(list(pool), 2)
            a1 = action_embed(t1.keyboard, t1.mouse)
            a2 = action_embed(t2.keyboard, t2.mouse)
            a_sim = float(F.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0)).item())
            if a_sim < 0.7:
                continue
            z1 = t1.latent_t.unsqueeze(0).to(self.device).float()
            z2 = t2.latent_t.unsqueeze(0).to(self.device).float()
            kb1, ms1 = _mean_action(t1.keyboard, t1.mouse)
            kb2, ms2 = _mean_action(t2.keyboard, t2.mouse)
            e1 = pool_latent(
                self._predict(
                    z1, kb1.unsqueeze(0).to(self.device), ms1.unsqueeze(0).to(self.device)
                )
                - z1
            )
            e2 = pool_latent(
                self._predict(
                    z2, kb2.unsqueeze(0).to(self.device), ms2.unsqueeze(0).to(self.device)
                )
                - z2
            )
            # GT residuals
            g1 = pool_latent(t1.latent_tp1.float() - t1.latent_t.float())
            g2 = pool_latent(t2.latent_tp1.float() - t2.latent_t.float())
            pred_sim = float(F.cosine_similarity(e1, e2, dim=-1).mean().item())
            gt_sim = float(F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item())
            # Score: how well predicted effect-similarity tracks GT
            sims.append(1.0 / (1.0 + abs(pred_sim - gt_sim)))
        if not sims:
            return 0.0
        return float(sum(sims) / len(sims))

    @torch.no_grad()
    def long_horizon(
        self,
        chains: Sequence[Sequence[Transition]],
    ) -> float:
        """Multi-step rollout consistency; returns 1/(1+mean_drift)."""
        if not chains:
            return 0.0
        drifts = []
        for chain in chains:
            if len(chain) < 2:
                continue
            z = chain[0].latent_t.unsqueeze(0).to(self.device).float()
            total = 0.0
            for t in chain:
                kb, ms = _mean_action(t.keyboard, t.mouse)
                z = self._predict(
                    z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device)
                )
                gt = t.latent_tp1.unsqueeze(0).to(self.device).float()
                total += float(F.mse_loss(z, gt).item())
            drifts.append(total / len(chain))
        if not drifts:
            return 0.0
        mean_d = sum(drifts) / len(drifts)
        return float(1.0 / (1.0 + mean_d))

    @torch.no_grad()
    def holdout_transfer(
        self,
        train_pool: Sequence[Transition],
        holdout_pool: Sequence[Transition],
    ) -> float:
        """Transfer dimension: predict holdout transitions after seeing train-like ones.

        Score = 1/(1+mse) on holdout; higher is better.
        """
        if not holdout_pool:
            return 0.0
        # lightly calibrate scale on train
        train_err = []
        for t in list(train_pool)[: min(16, len(train_pool))]:
            z = t.latent_t.unsqueeze(0).to(self.device).float()
            kb, ms = _mean_action(t.keyboard, t.mouse)
            pred = self._predict(z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device))
            gt = t.latent_tp1.unsqueeze(0).to(self.device).float()
            train_err.append(float(F.mse_loss(pred, gt).item()))
        hold_err = []
        for t in list(holdout_pool)[: min(32, len(holdout_pool))]:
            z = t.latent_t.unsqueeze(0).to(self.device).float()
            kb, ms = _mean_action(t.keyboard, t.mouse)
            pred = self._predict(z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device))
            gt = t.latent_tp1.unsqueeze(0).to(self.device).float()
            hold_err.append(float(F.mse_loss(pred, gt).item()))
        if not hold_err:
            return 0.0
        mean_h = sum(hold_err) / len(hold_err)
        # optional: penalize if holdout much worse than train
        if train_err:
            mean_t = sum(train_err) / len(train_err)
            gap = max(0.0, mean_h - mean_t)
            return float(1.0 / (1.0 + mean_h + gap))
        return float(1.0 / (1.0 + mean_h))

    def probe_batch(
        self,
        batch: Sequence[Transition],
        pool: Sequence[Transition],
        chains: Optional[Sequence[Sequence[Transition]]] = None,
        interventions: Optional[Sequence[Any]] = None,
        holdout_pool: Optional[Sequence[Transition]] = None,
    ) -> ProbeResult:
        if interventions:
            causal, per = self.sandbox_action_causality(interventions)
            cf = self.sandbox_counterfactual(interventions)
        else:
            causal, per = self.action_causality(batch, pool)
            cf = self.counterfactual(batch, pool)
        transfer = self.state_transfer(pool if len(pool) >= 4 else batch)
        horizon = self.long_horizon(chains or [])
        scores = {
            "action_causality": causal,
            "counterfactual": cf,
            "state_transfer": transfer,
            "long_horizon": horizon,
        }
        if holdout_pool is not None:
            scores["holdout_transfer"] = self.holdout_transfer(pool, holdout_pool)
        details = {
            "n_batch": float(len(batch)),
            "n_pool": float(len(pool)),
            "n_chains": float(len(chains or [])),
            "n_interventions": float(len(interventions or [])),
            "n_holdout": float(len(holdout_pool or [])),
            "sandbox_grounded": 1.0 if interventions else 0.0,
        }
        return ProbeResult(scores=scores, details=details, per_item=per)

    @torch.no_grad()
    def sandbox_action_causality(self, pairs: Sequence[Any]) -> Tuple[float, List[Dict[str, float]]]:
        """Use real Env outcomes under two actions from (approx) same state."""
        if not pairs:
            return 0.0, []
        scores, per = [], []
        for p in pairs:
            z = p.latent_t.unsqueeze(0).to(self.device).float()
            kb_a, ms_a = _mean_action(p.kb_a, p.ms_a)
            kb_b, ms_b = _mean_action(p.kb_b, p.ms_b)
            pred_a = self._predict(z, kb_a.unsqueeze(0).to(self.device), ms_a.unsqueeze(0).to(self.device))
            pred_b = self._predict(z, kb_b.unsqueeze(0).to(self.device), ms_b.unsqueeze(0).to(self.device))
            gt_a = p.latent_a.unsqueeze(0).to(self.device).float()
            gt_b = p.latent_b.unsqueeze(0).to(self.device).float()
            err_aa = float(F.mse_loss(pred_a, gt_a).item())
            err_ab = float(F.mse_loss(pred_a, gt_b).item())
            err_bb = float(F.mse_loss(pred_b, gt_b).item())
            err_ba = float(F.mse_loss(pred_b, gt_a).item())
            # matched action should beat cross pairing
            match = 0.5 * float(err_aa + self.causal_margin <= err_ab) + 0.5 * float(
                err_bb + self.causal_margin <= err_ba
            )
            sep = float(F.mse_loss(pred_a, pred_b).item())
            gt_sep = float(F.mse_loss(gt_a, gt_b).item())
            ctrl = 1.0 / (1.0 + abs(sep - gt_sep) / (gt_sep + 1e-4))
            score = 0.6 * match + 0.4 * ctrl
            scores.append(score)
            per.append({
                "action_causality": score,
                "err_aa": err_aa,
                "err_bb": err_bb,
                "match": match,
                "ctrl": ctrl,
            })
        return float(sum(scores) / len(scores)), per

    @torch.no_grad()
    def sandbox_counterfactual(self, pairs: Sequence[Any]) -> float:
        """Env produces different futures → model should too."""
        if not pairs:
            return 0.0
        scores = []
        for p in pairs:
            z = p.latent_t.unsqueeze(0).to(self.device).float()
            kb_a, ms_a = _mean_action(p.kb_a, p.ms_a)
            kb_b, ms_b = _mean_action(p.kb_b, p.ms_b)
            pred_a = self._predict(z, kb_a.unsqueeze(0).to(self.device), ms_a.unsqueeze(0).to(self.device))
            pred_b = self._predict(z, kb_b.unsqueeze(0).to(self.device), ms_b.unsqueeze(0).to(self.device))
            pred_sep = float(F.mse_loss(pred_a, pred_b).item())
            gt_sep = float(F.mse_loss(p.latent_a.float(), p.latent_b.float()).item())
            if gt_sep < 1e-8:
                # Env futures identical — model should also not hallucinate large gaps
                scores.append(float(1.0 / (1.0 + pred_sep * 10.0)))
            else:
                scores.append(float(1.0 / (1.0 + abs(pred_sep - gt_sep) / (gt_sep + 1e-4))))
        return float(sum(scores) / len(scores))


def build_horizon_chains(
    transitions: Sequence[Transition],
    horizon: int,
    max_chains: int = 16,
) -> List[List[Transition]]:
    """Greedy chains from consecutive block indices when meta has b0."""
    by_clip: Dict[int, List[Transition]] = {}
    # Without clip ids, group by ascending b0 in meta when present
    indexed = [(i, t) for i, t in enumerate(transitions) if "b0" in (t.meta or {})]
    if not indexed:
        # fallback: random short chains from pool
        out = []
        pool = list(transitions)
        for _ in range(min(max_chains, max(1, len(pool) // max(horizon, 1)))):
            if len(pool) < horizon:
                break
            out.append(random.sample(pool, horizon))
        return out

    indexed.sort(key=lambda x: x[1].meta.get("b0", 0))
    # Approximate chains by sorting global b0 (works within single-clip heavy buffers)
    sorted_t = [t for _, t in indexed]
    chains: List[List[Transition]] = []
    i = 0
    while i + horizon <= len(sorted_t) and len(chains) < max_chains:
        chunk = sorted_t[i : i + horizon]
        b0s = [t.meta.get("b0", -1) for t in chunk]
        if all(
            isinstance(b0s[j], int)
            and isinstance(b0s[j + 1], int)
            and b0s[j + 1] > b0s[j]
            for j in range(len(b0s) - 1)
        ):
            chains.append(chunk)
        i += horizon
    if not chains:
        # fallback
        for i in range(0, min(len(sorted_t) - horizon + 1, max_chains * horizon), horizon):
            chains.append(sorted_t[i : i + horizon])
    return chains
