#!/usr/bin/env python3
"""DiaWM probe policy: VLA as adversarial crack-finder for the WM.

r_probe = Uncertainty(F) + η · ‖g‖ + γ · abstain

  g  — cognitive gap vector (probe failures / novelty / unknown families)
  Training-time only: GRPO maximises r_probe in probe mode so π seeks
  states where the WM does *not* understand. Deploy drops this module.

Also: intervention ranking D_ij vs sandbox/buffer D̃_ij (sensitivity, not SEM).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from curriculum.knowledge_memory import WorldKnowledgeMemory, action_embed, pool_latent
from curriculum.probes import WorldModelProber, build_horizon_chains
from curriculum.replay_buffer import Transition
from curriculum.sandbox_experience import ACTION_NAMES, preset_mg2_action


GAP_DIM = 8  # [1-causal, 1-cf, 1-transfer, 1-horizon, uncert, novelty, abstain, unknown_fam]


def cognitive_gap_vector(
    probe_scores: Dict[str, float],
    *,
    uncert: float = 0.0,
    novelty: float = 0.0,
    abstain: float = 0.0,
    family_unknown: float = 0.0,
) -> np.ndarray:
    """g ∈ R^d: large = WM cognitive hole at this (s,a) neighborhood."""
    g = np.array([
        1.0 - float(probe_scores.get("action_causality", probe_scores.get("probe/action_causality", 0.0))),
        1.0 - float(probe_scores.get("counterfactual", probe_scores.get("probe/counterfactual", 0.0))),
        1.0 - float(probe_scores.get("state_transfer", probe_scores.get("probe/state_transfer", 0.0))),
        1.0 - float(probe_scores.get("long_horizon", probe_scores.get("probe/long_horizon", 0.0))),
        float(np.clip(uncert, 0.0, 1.0)),
        float(np.clip(novelty, 0.0, 1.0)),
        float(np.clip(abstain, 0.0, 1.0)),
        float(np.clip(family_unknown, 0.0, 1.0)),
    ], dtype=np.float32)
    return g


def intervention_ranking_score(
    wm: torch.nn.Module,
    z: torch.Tensor,
    z_next: torch.Tensor,
    kb: torch.Tensor,
    ms: torch.Tensor,
    *,
    device: torch.device,
    n_alt: int = 4,
) -> float:
    """Ranking accuracy: observed a should be closer to z' than random a' (0–1)."""
    wm.eval()
    names = [n for n in ACTION_NAMES if n != "noop"][: max(2, n_alt)]
    with torch.no_grad():
        zz = z.unsqueeze(0).to(device).float() if z.ndim == 4 else z.to(device).float()
        if zz.ndim == 4:
            zz = zz.unsqueeze(0)
        z1 = z_next.unsqueeze(0).to(device).float() if z_next.ndim == 4 else z_next.to(device).float()
        if z1.ndim == 4:
            z1 = z1.unsqueeze(0)
        k = kb.mean(0) if kb.ndim == 2 else kb
        m = ms.mean(0) if ms.ndim == 2 else ms
        pred = wm(zz, k.unsqueeze(0).to(device).float(), m.unsqueeze(0).to(device).float())
        err_obs = float(F.mse_loss(pred, z1).item())
        wins = 0
        for name in names:
            kb2, ms2 = preset_mg2_action(name)
            pred2 = wm(zz, kb2.unsqueeze(0).to(device).float(), ms2.unsqueeze(0).to(device).float())
            err_alt = float(F.mse_loss(pred2, z1).item())
            if err_obs + 1e-6 <= err_alt:
                wins += 1
        return float(wins / max(1, len(names)))


def score_probe_episode(
    wm: torch.nn.Module,
    transitions: Sequence[Transition],
    memory: Optional[WorldKnowledgeMemory] = None,
    *,
    device: Optional[torch.device] = None,
    uncert: float = 0.0,
) -> Dict[str, Any]:
    """Return r_probe ∈ [0,1] (higher = better crack-finding) + gap vector."""
    device = device or torch.device("cpu")
    empty = {
        "r_probe": 0.0,
        "gap_norm": 0.0,
        "gap": [0.0] * GAP_DIM,
        "ranking": 0.0,
        "abstain": 0.0,
        "novelty": 0.0,
        "n_trans": 0.0,
    }
    if len(transitions) < 2:
        empty["r_probe"] = float(np.clip(uncert, 0.0, 1.0))
        return empty

    prober = WorldModelProber(model=wm, device=device, horizon=3, n_intervene=3)
    chains = build_horizon_chains(list(transitions), horizon=3, max_chains=min(8, len(transitions)))
    snap = prober.probe_batch(list(transitions), list(transitions), chains=chains, interventions=None)
    scores = dict(snap.scores or {})

    novs, absn, ranks = [], [], []
    for tr in list(transitions)[:12]:
        c = pool_latent(tr.latent_t).cpu().reshape(-1)
        a = action_embed(tr.keyboard, tr.mouse).cpu().reshape(-1)
        if memory is not None and len(memory) > 0:
            novs.append(float(memory.novelty(c, a)))
            absn.append(float(memory.abstain_score(c, a)))
        else:
            novs.append(1.0)
            absn.append(1.0)
        try:
            ranks.append(intervention_ranking_score(
                wm, tr.latent_t, tr.latent_tp1, tr.keyboard, tr.mouse, device=device,
            ))
        except Exception:
            continue
    novelty = float(np.mean(novs)) if novs else 1.0
    abstain = float(np.mean(absn)) if absn else 1.0
    ranking = float(np.mean(ranks)) if ranks else 0.0
    fam_unk = 0.0
    if memory is not None:
        fam_unk = float(memory.coverage_report().get("family_unknown", 1.0))

    g = cognitive_gap_vector(
        scores, uncert=uncert, novelty=novelty, abstain=abstain, family_unknown=fam_unk,
    )
    gap_norm = float(np.clip(np.linalg.norm(g) / np.sqrt(GAP_DIM), 0.0, 1.0))
    # Probe reward: seek holes (high gap/uncert/abstain) AND expose bad ranking
    inv_rank = 1.0 - ranking
    r_probe = float(np.clip(
        0.35 * float(np.clip(uncert, 0.0, 1.0))
        + 0.35 * gap_norm
        + 0.20 * abstain
        + 0.10 * inv_rank,
        0.0, 1.0,
    ))
    return {
        "r_probe": r_probe,
        "gap_norm": gap_norm,
        "gap": [float(x) for x in g.tolist()],
        "ranking": ranking,
        "abstain": abstain,
        "novelty": novelty,
        "family_unknown": fam_unk,
        "probe/action_causality": float(scores.get("action_causality", 0.0)),
        "probe/counterfactual": float(scores.get("counterfactual", 0.0)),
        "probe/state_transfer": float(scores.get("state_transfer", 0.0)),
        "probe/long_horizon": float(scores.get("long_horizon", 0.0)),
        "n_trans": float(len(transitions)),
    }


def apply_memory_prior(
    pred: torch.Tensor,
    z: torch.Tensor,
    keyboard: torch.Tensor,
    mouse: torch.Tensor,
    memory: Optional[WorldKnowledgeMemory],
    *,
    alpha: float = 0.25,
    sim_thresh: float = 0.55,
) -> torch.Tensor:
    """Inject retrieved E* as residual prior bias (RP §3.3 readout)."""
    if memory is None or len(memory) == 0 or alpha <= 0:
        return pred
    out = pred.clone()
    for i in range(pred.shape[0]):
        zi = z[i]
        kb_i = keyboard[i] if keyboard.ndim >= 2 else keyboard
        ms_i = mouse[i] if mouse.ndim >= 2 else mouse
        if kb_i.ndim == 2:
            kb_i = kb_i.mean(0)
        if ms_i.ndim == 2:
            ms_i = ms_i.mean(0)
        c = pool_latent(zi).detach().cpu().reshape(-1)
        a = action_embed(kb_i.detach().cpu(), ms_i.detach().cpu()).reshape(-1)
        nn = memory.nearest(c, a, topk=1)
        if not nn:
            continue
        it, sim = nn[0]
        if it.status == "revoked" or sim < sim_thresh:
            continue
        e = it.effect_t().to(pred.device, pred.dtype)
        d = min(e.numel(), pred.shape[1])
        bias = torch.zeros_like(out[i])
        view_shape = (d,) + (1,) * (out[i].ndim - 1)
        bias[:d] = e[:d].reshape(*view_shape)
        out[i] = out[i] + (float(alpha) * float(it.confidence) * float(sim)) * bias
    return out


def probe_focus_actions(memory: Optional[WorldKnowledgeMemory], k: int = 4) -> List[str]:
    """Actions the probe VLA should prefer (unknown / stale families)."""
    if memory is None or len(memory) == 0:
        return list(ACTION_NAMES[:k])
    cov = memory.coverage_report()
    per = cov.get("per_action") or {}
    ranked = sorted(
        per.items(),
        key=lambda kv: (float(kv[1].get("unknown", 0.0)) + 0.5 * float(kv[1].get("revoked", 0.0)),
                        -float(kv[1].get("mastered", 0.0))),
        reverse=True,
    )
    names = [n for n, _ in ranked if n != "noop"][:k]
    return names or list(ACTION_NAMES[:k])
