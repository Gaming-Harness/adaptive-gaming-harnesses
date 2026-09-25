#!/usr/bin/env python3
"""Fast test: VLA action → ActionProjection → WM(z, a_proj).

Does NOT reload 8B / sandbox (those were the slow parts).
Uses a synthetic MineStudio action chunk as stand-in for VLA output,
plus optional --from_text to parse real cold-start text.

Usage:
  cd project_root
  python -m curriculum.smoke_vla_action_to_wm
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from curriculum.action_codec import noop_action
from curriculum.action_projection import (
    ActionProjection,
    hard_codec_mg2,
    projection_align_loss,
)
from curriculum.frozen_wm import FrozenWorldModel, ToyDynamics
from curriculum.replay_buffer import MixedReplayBuffer


def fake_vla_chunk():
    """Simulate VLA MineStudio chunk (what parse_actions_text would return)."""
    acts = []
    for i in range(4):
        a = noop_action()
        a["forward"] = 1
        a["camera"] = [2.0 * i, -1.5]
        if i == 2:
            a["attack"] = 1
        acts.append(a)
    return acts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # --- 1) VLA-side actions (no 8B) ---
    actions = fake_vla_chunk()
    print(f"[1] VLA chunk len={len(actions)} forward={actions[0]['forward']} cam={actions[0]['camera']}")

    # --- 2) Projection bridge ---
    proj = ActionProjection().to(device)
    # quick align to hard codec so bridge is meaningful
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    for i in range(50):
        loss = projection_align_loss(proj, actions)
        opt.zero_grad(); loss.backward(); opt.step()
    kb_p, ms_p, cond = proj.from_minestudio_chunk(actions, device=device)
    kb_h, ms_h = hard_codec_mg2(actions)
    print(f"[2] proj kb={kb_p.detach().cpu().numpy().round(3)} "
          f"hard_mean={kb_h.mean(0).numpy().round(3)}")
    print(f"[2] proj mouse={ms_p.detach().cpu().numpy().round(3)} "
          f"hard_mean={ms_h.mean(0).numpy().round(3)}")
    print(f"[2] cond.shape={tuple(cond.shape)} align_loss={loss.item():.4f}")

    # --- 3) Feed projected action into WM ---
    toy_ckpt = "curriculum/outputs/toy_wm0.pt"
    wrap = FrozenWorldModel(
        backend="toy", device=str(device),
        toy_ckpt=toy_ckpt if os.path.exists(toy_ckpt) else None,
    )
    student = ToyDynamics().to(device)
    student.load_state_dict(wrap.toy.state_dict())

    buf = MixedReplayBuffer()
    n = buf.load_expert_pt_dir("data/mc_vpt_long45", max_clips=2, block_frames=3, stride=2)
    tr = buf.sample(1, prefer="expert")[0]
    z = tr.latent_t.unsqueeze(0).to(device).float()
    z1 = tr.latent_tp1.unsqueeze(0).to(device).float()

    # KEY: WM conditioned on *projected VLA action*, not expert action
    with torch.no_grad():
        z_hat_vla = wrap.toy(z, kb_p.detach(), ms_p.detach())
        # baseline: expert action
        kb_e = tr.keyboard.float()
        ms_e = tr.mouse.float()
        if kb_e.ndim == 2:
            kb_e, ms_e = kb_e.mean(0), ms_e.mean(0)
        z_hat_exp = wrap.toy(z, kb_e.unsqueeze(0).to(device), ms_e.unsqueeze(0).to(device))

    err_vla = float((z_hat_vla - z1).pow(2).mean().item())
    err_exp = float((z_hat_exp - z1).pow(2).mean().item())
    print(f"[3] WM(z, a_vla_proj) mse_to_s'={err_vla:.4f}")
    print(f"[3] WM(z, a_expert)    mse_to_s'={err_exp:.4f}")

    # one grad step: train WM on (z, a_vla_proj, z')  — on-policy style
    opt_w = torch.optim.Adam(student.parameters(), lr=1e-4)
    pred = student(z, kb_p.detach(), ms_p.detach())
    loss_w = F.mse_loss(pred, z1)
    opt_w.zero_grad(); loss_w.backward(); opt_w.step()
    print(f"[4] WM update on VLA-projected action | loss={loss_w.item():.4f}")

    # save bridge
    out = "curriculum/outputs/action_projection.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"projection": proj.state_dict(), "in_dim": proj.in_dim}, out)
    print(f"[5] saved {out}")
    print(f"\nPASS vla_action→projection→WM in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
