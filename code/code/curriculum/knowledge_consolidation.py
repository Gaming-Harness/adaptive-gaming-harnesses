#!/usr/bin/env python3
"""Knowledge Consolidation — Method + **positioning** for the Auto-WM RP.

Primary claim (do NOT sell as “beat Dreamer MSE”)
-------------------------------------------------
  Self-aware world model: probes draw a **knowledge boundary**;
  memory is a verified ops-manual with **revoke**; GRPO credit
  refuses connection-only shortcuts. Goal = coverage ↑, unknown
  detectable, not pixel-1% better.

Naming: probes are **intervention sensitivity**, not causal discovery
(no SEM / IV; confounders not identified).

Paradox (sandbox always available → why not plain online RL?)
-------------------------------------------------------------
  VLA already does online GRPO on sandbox. Probes mostly run in
  *latent* on already-collected τ (cheap a≠a' rollouts of F_φ),
  not extra env resets every step. Memory decides Skip vs Learn
  (don't retrain what is mastered; re-verify / revoke if stale).
  Extra sandbox interventions are optional grounding, not the
  main train loop. Simulator-scoped on purpose.

Eval vs train
-------------
  First-class contribution = evaluation protocol (probes, essence,
  visual GT vs ŝ, coverage/abstain).
  Train-side = credit assignment + curriculum gating, not a new PG.

--------------------------------------------------------------------------------
Formalism (latent space; Dreamer-style — NOT pixel long-horizon distance)
--------------------------------------------------------------------------------

Knowledge slot (differentiable Key–Value, NOT free-text)::

    m_i = (C_i, A_i, E_i, conf_i) ∈ M

      C_i = pool(z_t)           # condition / state abstractor  (Key part)
      A_i = pool([kb; mouse])   # action abstractor             (Key part)
      E_i = pool(z_{t+1}) - C_i # transition residual Δz        (Value)
      conf_i = Probe(τ)         # only written if probes pass

Extraction operator G (probe-gated compress)::

    m_i = G(τ, Probe) :=
        accept (C,A,E,conf)  if  conf ≥ τ_pass
        else discard / enqueue failure for intrinsic re-sampling

World-model update (when WorthLearn = LEARN)::

    L = L_dyn + λ_ic · L_IC + λ_mem · L_mem

      L_dyn = || F_φ(z_t, a_t) - z_{t+1} ||²

      L_IC  (Intervention Consistency): for contrast actions a, a'
            Δ̂ = F(z,a)-F(z,a'),  Δ* = z'(a)-z'(a')  (sandbox or buffer pair)
            L_IC = 1 - cos(Δ̂, Δ*)     # direction of effect, not just magnitude

      L_mem (slot readout consistency): retrieve nearest (C,A) → E
            L_mem = || pool(F(z,a)) - pool(z) - E ||² · conf

GRPO / RL interface (memory enters the *reward*, not the optimizer class)::

    r = r_env
        + λ_mse  · connection(F)          # pred-MSE unit
        + λ_xfer · essence(Probe)         # transfer − β·shortcut
        + λ_slot · align(τ, M)            # trajectory agrees with retrieved slots

    π ← GRPO(r)     # still GRPO; novelty is in credit / curriculum, not PG form

Intrinsic motivation (Self-Evolving)::

    WorthLearn(H) = U·N·V − λ Cost
    failed probes ↑ U and enqueue H → prioritize sandbox/VLA data that patches M

Chaos / long-horizon::

    Probes use *latent* distances + uncertainty ensemble, not raw pixel d_t.
    Prefer calibration of P(z_{t+k}) / uncertainty-gated horizon over single-point D.

Simulator scope:: Minecraft / MineStudio sandbox — resettable interventions.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from curriculum.knowledge_memory import (
    WorldKnowledgeMemory,
    action_embed,
    pool_latent,
)
from curriculum.replay_buffer import Transition
from curriculum.sandbox_experience import expand_action_window, preset_mg2_action


def slot_from_transition(tr: Transition) -> Dict[str, torch.Tensor]:
    """G's featurizer: τ → (C, A, E) in latent abstract space."""
    c = pool_latent(tr.latent_t).float().reshape(-1)
    a = action_embed(tr.keyboard, tr.mouse).float().reshape(-1)
    e = pool_latent(tr.latent_tp1).float().reshape(-1) - c
    return {"C": c, "A": a, "E": e}


def intervention_consistency_loss(
    wm: torch.nn.Module,
    batch: Sequence[Transition],
    *,
    device: torch.device,
    n_contrast: int = 2,
) -> torch.Tensor:
    """L_IC: align *direction* of Δ under action intervention (not just MSE).

    For each (z, a, z') sample an alternate action a' and require
      cos( F(z,a)-F(z,a'),  z'-F(z,a')_stopgrad_proxy )  high.

    When a true counterfactual z'(a') is unavailable (no sandbox pair), we use
    the residual target z'-z as the reference direction for a, and require
    F(z,a)-F(z,a') to be closer to that residual than a random direction —
    implemented as: maximize cos(F(z,a)-F(z,a'), z'-z) while a' ≠ a.
    """
    if len(batch) < 1:
        return torch.zeros((), device=device)

    def _z5(z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        if z.ndim == 4:
            return z
        if z.ndim == 3:
            return z.unsqueeze(1)
        return z

    losses = []
    alt_names = ["forward", "back", "left", "right", "turn_left", "turn_right", "noop"]
    for tr in batch:
        z = _z5(tr.latent_t).unsqueeze(0).to(device)
        z1 = _z5(tr.latent_tp1).unsqueeze(0).to(device)
        kb = tr.keyboard.mean(0) if tr.keyboard.ndim == 2 else tr.keyboard
        ms = tr.mouse.mean(0) if tr.mouse.ndim == 2 else tr.mouse
        kb = kb.float().unsqueeze(0).to(device)
        ms = ms.float().unsqueeze(0).to(device)
        pred_a = wm(z, kb, ms)
        delta_tgt = (z1 - z).mean(dim=(2, 3, 4))  # [1,C]
        if float(delta_tgt.norm()) < 1e-6:
            continue
        local = []
        for j in range(n_contrast):
            name = alt_names[(hash((id(tr), j)) + j) % len(alt_names)]
            kb2, ms2 = preset_mg2_action(name)
            # match action window length loosely via mean tick
            kb2 = kb2.float().unsqueeze(0).to(device)
            ms2 = ms2.float().unsqueeze(0).to(device)
            pred_b = wm(z, kb2, ms2)
            delta_hat = (pred_a - pred_b).mean(dim=(2, 3, 4))
            if float(delta_hat.norm()) < 1e-8:
                local.append(torch.tensor(1.0, device=device))
                continue
            cos = F.cosine_similarity(delta_hat, delta_tgt, dim=-1).mean()
            local.append(1.0 - cos)
        if local:
            losses.append(torch.stack(local).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def memory_consistency_loss(
    wm: torch.nn.Module,
    memory: WorldKnowledgeMemory,
    batch: Sequence[Transition],
    *,
    device: torch.device,
    sim_thresh: float = 0.55,
) -> torch.Tensor:
    """L_mem: if (C,A) retrieves a slot, F's residual should match E."""
    if len(memory) == 0 or len(batch) == 0:
        return torch.zeros((), device=device)

    def _z5(z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        if z.ndim == 4:
            return z
        if z.ndim == 3:
            return z.unsqueeze(1)
        return z

    losses = []
    for tr in batch:
        c = pool_latent(tr.latent_t).cpu().reshape(-1)
        a = action_embed(tr.keyboard, tr.mouse).cpu().reshape(-1)
        nn = memory.nearest(c, a, topk=1)
        if not nn:
            continue
        it, sim = nn[0]
        if it.status == "revoked" or sim < sim_thresh:
            continue
        z = _z5(tr.latent_t).unsqueeze(0).to(device)
        kb = tr.keyboard.mean(0) if tr.keyboard.ndim == 2 else tr.keyboard
        ms = tr.mouse.mean(0) if tr.mouse.ndim == 2 else tr.mouse
        kb = kb.float().unsqueeze(0).to(device)
        ms = ms.float().unsqueeze(0).to(device)
        pred = wm(z, kb, ms)
        e_hat = pool_latent(pred[0]) - pool_latent(z[0])
        e = it.effect_t().to(device)
        # pad/trim
        d = min(e_hat.numel(), e.numel())
        w = float(it.confidence) * float(sim)
        losses.append(w * F.mse_loss(e_hat.reshape(-1)[:d], e.reshape(-1)[:d]))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def consolidate_wm_loss(
    wm: torch.nn.Module,
    batch: Sequence[Transition],
    memory: Optional[WorldKnowledgeMemory] = None,
    *,
    device: torch.device,
    lambda_ic: float = 0.3,
    lambda_mem: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Full Method loss: L_dyn + λ_ic L_IC + λ_mem L_mem."""
    if not batch:
        z = torch.zeros((), device=device)
        return z, {"loss/dyn": 0.0, "loss/ic": 0.0, "loss/mem": 0.0, "loss/total": 0.0}

    def _z5(z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        if z.ndim == 4:
            return z
        if z.ndim == 3:
            return z.unsqueeze(1)
        return z

    zs = torch.stack([_z5(t.latent_t) for t in batch]).to(device)
    z1 = torch.stack([_z5(t.latent_tp1) for t in batch]).to(device)
    kbs, mss = [], []
    for t in batch:
        kb = t.keyboard.mean(0) if t.keyboard.ndim == 2 else t.keyboard
        ms = t.mouse.mean(0) if t.mouse.ndim == 2 else t.mouse
        kbs.append(kb.float())
        mss.append(ms.float())
    kb = torch.stack(kbs).to(device)
    ms = torch.stack(mss).to(device)
    pred = wm(zs, kb, ms)
    try:
        from curriculum.probe_policy import apply_memory_prior
        pred = apply_memory_prior(pred, zs, kb, ms, memory, alpha=0.25)
    except Exception:
        pass
    l_dyn = F.mse_loss(pred, z1)
    l_ic = intervention_consistency_loss(wm, batch, device=device)
    l_mem = (
        memory_consistency_loss(wm, memory, batch, device=device)
        if memory is not None else torch.zeros((), device=device)
    )
    total = l_dyn + float(lambda_ic) * l_ic + float(lambda_mem) * l_mem
    stats = {
        "loss/dyn": float(l_dyn.detach().item()),
        "loss/ic": float(l_ic.detach().item()),
        "loss/mem": float(l_mem.detach().item()),
        "loss/total": float(total.detach().item()),
        "lambda_ic": float(lambda_ic),
        "lambda_mem": float(lambda_mem),
    }
    return total, stats


def memory_slot_alignment(
    memory: WorldKnowledgeMemory,
    transitions: Sequence[Transition],
    *,
    sim_thresh: float = 0.55,
) -> Dict[str, float]:
    """GRPO bonus unit: how well τ agrees with retrieved slots (no grad).

    align ∈ [-1, 1]: + if residual matches E · conf; − if disagrees on a
    retrieved high-conf slot (connection-like mismatch).
    """
    if len(memory) == 0 or not transitions:
        return {"slot_align": 0.0, "n_hit": 0.0, "n_miss": 0.0}
    hits, scores = 0, []
    for tr in transitions:
        c = pool_latent(tr.latent_t).cpu().reshape(-1)
        a = action_embed(tr.keyboard, tr.mouse).cpu().reshape(-1)
        e = pool_latent(tr.latent_tp1).cpu().reshape(-1) - c
        nn = memory.nearest(c, a, topk=1)
        if not nn:
            continue
        it, sim = nn[0]
        if it.status == "revoked" or sim < sim_thresh:
            continue
        hits += 1
        e_m = it.effect_t()
        d = min(e.numel(), e_m.numel())
        cos = float(F.cosine_similarity(
            e.reshape(1, -1)[:, :d], e_m.reshape(1, -1)[:, :d], dim=-1
        ).item())
        scores.append(cos * float(it.confidence) * float(sim))
    if not scores:
        return {"slot_align": 0.0, "n_hit": 0.0, "n_miss": float(len(transitions))}
    return {
        "slot_align": float(sum(scores) / len(scores)),
        "n_hit": float(hits),
        "n_miss": float(max(0, len(transitions) - hits)),
    }


def consolidate_from_probes(
    memory: WorldKnowledgeMemory,
    transitions: Sequence[Transition],
    probe_scores: Dict[str, float],
    confidence: float,
) -> List[str]:
    """Explicit G(τ, Probe): write slots only when probes pass (caller gates)."""
    ids = []
    for tr in transitions:
        item = memory.propose(
            tr.latent_t,
            tr.keyboard,
            tr.mouse,
            tr.latent_tp1,
            probe_scores=probe_scores,
            confidence=confidence,
            meta={**(tr.meta or {}), "formal": "KnowledgeSlot(C,A,E)"},
        )
        if item is not None:
            ids.append(item.item_id)
    return ids
