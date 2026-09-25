"""Policy transfer vs connection probes for Qwen VLA GRPO.

GRPO env reward (and even WM pred-MSE bonus) can be gamed by learning
*connection patterns*: visual co-occurrence / frequent action sequences.

This module scores a VLA sandbox episode on whether the implied dynamics
look like **true action-conditioned transfer**:

  connection  ≈ how well F fits the observed (s,a,s') trajectory (MSE bonus)
  transfer    ≈ causality + counterfactual + state-transfer + long-horizon
  essence     = transfer − β · max(0, connection − transfer)

High essence → policy experience is worth GRPO credit / WM Learn.
Low essence  → likely shortcut; down-weight or Skip.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from curriculum.action_codec import chunk_to_mg2, parse_actions_text
from curriculum.probes import WorldModelProber, build_horizon_chains
from curriculum.replay_buffer import Transition
from curriculum.sandbox_experience import expand_action_window
from curriculum.wm_reward_bridge import cheap_latent_from_frames


HOLD_OUT_ACTIONS = {
    "turn_left", "turn_right", "look_up", "look_down",
    "forward_left", "forward_right", "camera", "attack", "use", "jump",
}


def _mg2_name(kb: torch.Tensor, ms: torch.Tensor) -> str:
    kb = kb.mean(0) if kb.ndim == 2 else kb
    ms = ms.mean(0) if ms.ndim == 2 else ms
    key_i = int(kb.argmax().item()) if float(kb.abs().sum()) > 0.2 else -1
    keys = {0: "forward", 1: "back", 2: "left", 3: "right"}
    name = keys.get(key_i, "noop")
    if float(ms.abs().sum()) > 0.15:
        pitch, yaw = float(ms[0]), float(ms[1])
        if abs(yaw) >= abs(pitch) and abs(yaw) > 0.1:
            name = "turn_right" if yaw > 0 else "turn_left"
        elif abs(pitch) > 0.1:
            name = "look_down" if pitch > 0 else "look_up"
        if key_i in (0,) and abs(yaw) > 0.1:
            name = "forward_right" if yaw > 0 else "forward_left"
    return name


def episode_to_transitions(
    image_paths: Sequence[str],
    action_texts: Sequence[str],
    *,
    nfpb: int = 3,
    max_turns: int = 24,
) -> List[Transition]:
    """VLA POV + action text → WM transitions (cheap latent + pretrained_wm actions)."""
    from PIL import Image

    frames: List[np.ndarray] = []
    for p in image_paths:
        if not p or not os.path.isfile(str(p)):
            continue
        try:
            frames.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
        except Exception:
            continue
    n_turns = min(len(action_texts), max(0, len(frames) - 1), max_turns)
    if n_turns <= 0 or len(frames) < nfpb + 1:
        return []

    out: List[Transition] = []
    for t in range(n_turns):
        end = min(t + nfpb + 1, len(frames))
        start = end - (nfpb + 1)
        if start < 0:
            continue
        window = frames[start:end]
        z_t = cheap_latent_from_frames(window[:-1], nfpb=nfpb)
        z_tp1 = cheap_latent_from_frames(window[1:], nfpb=nfpb)
        acts, _ = parse_actions_text(action_texts[t] if t < len(action_texts) else "")
        if acts:
            kb, ms = chunk_to_mg2(acts)
        else:
            kb = torch.zeros(max(1, 4 * (nfpb - 1) + 1), 4)
            ms = torch.zeros(kb.shape[0], 2)
        kb_w, ms_w = expand_action_window(kb.mean(0), ms.mean(0), nfpb)
        name = _mg2_name(kb, ms)
        out.append(Transition(
            latent_t=z_t,
            keyboard=kb_w,
            mouse=ms_w,
            latent_tp1=z_tp1,
            reward=0.0,
            source="policy",
            meta={"action": name, "turn": t, "vla": True},
        ))
    return out


def score_policy_episode(
    wm: torch.nn.Module,
    image_paths: Sequence[str],
    action_texts: Sequence[str],
    *,
    device: Optional[torch.device] = None,
    connection_unit: float = 0.0,
    beta: float = 0.7,
    nfpb: int = 3,
    max_turns: int = 16,
) -> Dict[str, Any]:
    """Return connection / transfer / essence diagnostics for one VLA episode."""
    device = device or torch.device("cpu")
    trans = episode_to_transitions(
        image_paths, action_texts, nfpb=nfpb, max_turns=max_turns,
    )
    empty = {
        "connection": float(connection_unit),
        "transfer": 0.0,
        "essence": 0.0,
        "probe/action_causality": 0.0,
        "probe/counterfactual": 0.0,
        "probe/state_transfer": 0.0,
        "probe/long_horizon": 0.0,
        "n_trans": float(len(trans)),
        "holdout_action_frac": 0.0,
    }
    if len(trans) < 2:
        empty["essence"] = float(-abs(connection_unit) * 0.25)
        return empty

    prober = WorldModelProber(model=wm, device=device, horizon=3, n_intervene=3)
    chains = build_horizon_chains(trans, horizon=3, max_chains=min(8, len(trans)))
    snap = prober.probe_batch(trans, trans, chains=chains, interventions=None)
    causal = float(snap.scores.get("action_causality", 0.0))
    cf = float(snap.scores.get("counterfactual", 0.0))
    st = float(snap.scores.get("state_transfer", 0.0))
    hz = float(snap.scores.get("long_horizon", 0.0))
    transfer = float(0.35 * causal + 0.35 * cf + 0.20 * st + 0.10 * hz)

    hold_n = sum(
        1 for t in trans
        if str((t.meta or {}).get("action", "")) in HOLD_OUT_ACTIONS
    )
    hold_frac = float(hold_n / max(1, len(trans)))
    # Holdout-family actions in the policy trajectory are evidence of broader transfer
    transfer = float(min(1.0, transfer + 0.1 * hold_frac))

    conn = float(connection_unit)
    shortcut = max(0.0, conn - transfer)
    essence = float(np.clip(transfer - float(beta) * shortcut, -1.0, 1.0))

    return {
        "connection": conn,
        "transfer": transfer,
        "essence": essence,
        "probe/action_causality": causal,
        "probe/counterfactual": cf,
        "probe/state_transfer": st,
        "probe/long_horizon": hz,
        "n_trans": float(len(trans)),
        "holdout_action_frac": hold_frac,
        "shortcut": float(shortcut),
    }
