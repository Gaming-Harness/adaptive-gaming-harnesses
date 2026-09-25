#!/usr/bin/env python3
"""Local GPU smoke: sandbox → (stub|hf) VLA → one WM update.

Usage:
  cd project_root
  PYTHONPATH=../../ares:$PYTHONPATH \\
    python -m curriculum.smoke_local_gpu --mode stub
  ... --mode hf   # loads collaborator_b 8B on local H200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ARES = os.path.abspath(os.path.join(_ROOT, "..", "..", "ares"))
sys.path.insert(0, _ROOT)
if _ARES not in sys.path:
    sys.path.insert(0, _ARES)
os.chdir(_ROOT)


def test_sandbox(task_dir: str) -> object:
    from curriculum.sandbox_bridge import MineStudioSandbox, SandboxConfig, extract_pil
    cfg = SandboxConfig.from_env()
    print(f"[1] sandbox endpoint={cfg.endpoint} token_len={len(cfg.token)}")
    sb = MineStudioSandbox(cfg, img_save_dir="curriculum/outputs/smoke_local/sandbox_images")
    tasks = sorted(
        os.path.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith(".json")
    )
    task = tasks[0] if tasks else None
    print(f"[1] reset task={task}")
    t0 = time.time()
    obs = sb.reset(task_config_file_path=task)
    img = extract_pil(obs)
    print(f"[1] reset ok in {time.time()-t0:.1f}s | img={None if img is None else img.size} | keys={list(obs)[:8] if isinstance(obs, dict) else type(obs)}")
    return sb, img, obs


def test_vla(mode: str, img, instruction: str):
    from curriculum.qwen_vla import QwenVLAConfig, QwenVLAPolicy
    cfg = QwenVLAConfig(
        mode=mode,
        device="cuda",
        dtype="bfloat16",
        max_new_tokens=128,
        instruction=instruction,
    )
    print(f"[2] load VLA mode={mode}")
    t0 = time.time()
    vla = QwenVLAPolicy(cfg)
    print(f"[2] loaded in {time.time()-t0:.1f}s")
    t1 = time.time()
    out = vla.act_image(img, instruction=instruction)
    print(f"[2] act in {time.time()-t1:.1f}s | n_actions={len(out['actions'])} | done={out['done']}")
    print(f"[2] text[:200]={out['text'][:200]!r}")
    print(f"[2] kb.shape={tuple(out['keyboard'].shape)} mouse.shape={tuple(out['mouse'].shape)}")
    return vla, out


def test_step_and_wm(sb, actions, toy_wm_ckpt: str):
    print("[3] sandbox.step(action_chunk)")
    t0 = time.time()
    obs2 = sb.step(actions if len(actions) > 1 else actions[0])
    from curriculum.sandbox_bridge import extract_pil
    img2 = extract_pil(obs2)
    r = obs2.get("reward", None) if isinstance(obs2, dict) else None
    print(f"[3] step ok in {time.time()-t0:.1f}s | img={None if img2 is None else img2.size} | reward={r}")

    import torch
    from curriculum.frozen_wm import FrozenWorldModel, ToyDynamics
    from curriculum.replay_buffer import MixedReplayBuffer, Transition
    import torch.nn.functional as F

    print("[4] toy WM one update on random expert + fake policy trans")
    wrap = FrozenWorldModel(backend="toy", device="cuda", toy_ckpt=toy_wm_ckpt if os.path.exists(toy_wm_ckpt) else None)
    student = ToyDynamics().cuda()
    student.load_state_dict(wrap.toy.state_dict())
    opt = torch.optim.Adam(student.parameters(), lr=1e-4)
    buf = MixedReplayBuffer()
    n = buf.load_expert_pt_dir("data/mc_vpt_long45", max_clips=2, block_frames=3, stride=2)
    print(f"[4] expert transitions={n}")
    batch = buf.sample(min(8, n), prefer="expert")
    z = torch.stack([t.latent_t.float() for t in batch]).cuda()
    z1 = torch.stack([t.latent_tp1.float() for t in batch]).cuda()
    kb = torch.stack([(t.keyboard.float().mean(0) if t.keyboard.ndim == 2 else t.keyboard.float()) for t in batch]).cuda()
    ms = torch.stack([(t.mouse.float().mean(0) if t.mouse.ndim == 2 else t.mouse.float()) for t in batch]).cuda()
    pred = student(z, kb, ms)
    loss = F.mse_loss(pred, z1)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"[4] wm loss={loss.item():.4f}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stub", "hf"], default="stub")
    ap.add_argument("--task_dir", default="/path/to/openha_tasks")
    ap.add_argument("--toy_wm_ckpt", default="curriculum/outputs/toy_wm0.pt")
    ap.add_argument("--skip_sandbox", action="store_true")
    args = ap.parse_args()
    os.makedirs("curriculum/outputs/smoke_local", exist_ok=True)

    ok = {"sandbox": False, "vla": False, "wm": False}
    try:
        if args.skip_sandbox:
            from PIL import Image
            import numpy as np
            img = Image.fromarray(np.zeros((360, 640, 3), dtype=np.uint8))
            sb = None
            instruction = "Mine oak log."
        else:
            sb, img, obs = test_sandbox(args.task_dir)
            ok["sandbox"] = True
            instruction = "Mine oak log."
            if isinstance(obs, dict):
                instruction = obs.get("instruction") or obs.get("task_text") or instruction
            if img is None:
                raise RuntimeError("no POV image from sandbox")

        vla, out = test_vla(args.mode, img, instruction)
        ok["vla"] = True

        if sb is not None:
            test_step_and_wm(sb, out["actions"], args.toy_wm_ckpt)
            ok["wm"] = True
            sb.close()
        else:
            # wm-only without sandbox step
            import torch, torch.nn.functional as F
            from curriculum.frozen_wm import FrozenWorldModel, ToyDynamics
            from curriculum.replay_buffer import MixedReplayBuffer
            wrap = FrozenWorldModel(backend="toy", device="cuda", toy_ckpt=args.toy_wm_ckpt if os.path.exists(args.toy_wm_ckpt) else None)
            student = ToyDynamics().cuda()
            student.load_state_dict(wrap.toy.state_dict())
            buf = MixedReplayBuffer()
            n = buf.load_expert_pt_dir("data/mc_vpt_long45", max_clips=2, block_frames=3, stride=2)
            batch = buf.sample(min(8, n), prefer="expert")
            z = torch.stack([t.latent_t.float() for t in batch]).cuda()
            z1 = torch.stack([t.latent_tp1.float() for t in batch]).cuda()
            kb = torch.stack([(t.keyboard.float().mean(0) if t.keyboard.ndim == 2 else t.keyboard.float()) for t in batch]).cuda()
            ms = torch.stack([(t.mouse.float().mean(0) if t.mouse.ndim == 2 else t.mouse.float()) for t in batch]).cuda()
            loss = F.mse_loss(student(z, kb, ms), z1)
            print(f"[4] wm loss (no sandbox)={loss.item():.4f}")
            ok["wm"] = True

        print("\n=== SMOKE RESULT ===")
        print(ok)
        if all(ok.values()) or (args.skip_sandbox and ok["vla"] and ok["wm"]):
            print("PASS")
            return 0
        print("PARTIAL")
        return 1
    except Exception:
        traceback.print_exc()
        print("\n=== SMOKE RESULT ===")
        print(ok)
        print("FAIL")
        return 2


if __name__ == "__main__":
    sys.exit(main())
