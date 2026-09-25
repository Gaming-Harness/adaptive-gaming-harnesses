#!/usr/bin/env python3
"""Eval diagnostic: sandbox frames vs WM-generated frames (visual metrics).

Task success still comes from sandbox. This module answers: given the same
actions, how close is F_φ(s,a) to the real next observation?

Cheap path (current GRPO): invert RGB channels of cheap latent → PSNR/SSIM/MAE.
VAE path (optional): Wan decode when encoder is loaded.

Usage:
  python -m curriculum.eval_wm_visual \\
    --traj_dir .../eval_aligned_zh_vla_fresh_turn_recent10_full37_repro/trajectories \\
    --ckpt_dir curriculum/outputs/coevolve_vla_auto \\
    --out curriculum/outputs/eval_wm_visual_qwen

  python -m curriculum.run_curriculum smoke_eval_wm_visual
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HOME = "/path/to/project"
sys.path.insert(0, _ROOT)


def _assert_write_home(path: str) -> str:
    abs_p = os.path.abspath(path if os.path.isabs(path) else os.path.join(_ROOT, path))
    if not (abs_p == _HOME or abs_p.startswith(_HOME + os.sep)):
        raise RuntimeError(f"[safety] write path must be under {_HOME}, got {abs_p}")
    return abs_p


def wrap_action_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "<actions> no_op ; no_op ; no_op ; no_op</actions>"
    if "<actions>" in t.lower():
        return t
    return f"<actions> {t} </actions>"


def cheap_latent_to_rgb(z, *, frame_idx: int = -1) -> np.ndarray:
    """Invert cheap_latent_from_frames RGB channels → uint8 HxW3 (44x80)."""
    import torch

    if isinstance(z, torch.Tensor):
        z = z.detach().float().cpu()
        if z.ndim == 5:
            z = z[0]
        arr = z.numpy()
    else:
        arr = np.asarray(z, dtype=np.float32)
    # arr: [16, f, H, W]
    if arr.ndim != 4:
        raise ValueError(f"expected [16,f,H,W], got {arr.shape}")
    fi = frame_idx if frame_idx >= 0 else arr.shape[1] - 1
    rgb = np.clip(arr[:3, fi], 0.0, 1.0)
    rgb = np.transpose(rgb, (1, 2, 0))
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def resize_rgb(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    from PIL import Image

    h, w = hw
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Channel-mean SSIM on uint8 RGB (no extra deps)."""
    x = pred.astype(np.float64)
    y = gt.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    scores = []
    for c in range(x.shape[-1]):
        a, b = x[..., c], y[..., c]
        mu_a, mu_b = a.mean(), b.mean()
        var_a = a.var()
        var_b = b.var()
        cov = float(((a - mu_a) * (b - mu_b)).mean())
        num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
        scores.append(float(num / max(den, 1e-12)))
    return float(np.mean(scores))


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))) / 255.0)


def _list_traj_dirs(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, "meta.json")) or os.path.isfile(os.path.join(d, "meta_lite.json")):
            out.append(d)
    return out


def load_episode(ep_dir: str) -> Dict[str, Any]:
    lite = os.path.join(ep_dir, "meta_lite.json")
    meta_p = os.path.join(ep_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.isfile(lite):
        meta = json.load(open(lite))
    elif os.path.isfile(meta_p):
        # ARES dump or huge eval meta — avoid loading giant blobs when possible
        try:
            sz = os.path.getsize(meta_p)
        except OSError:
            sz = 0
        if sz > 8_000_000:
            # still need frames/actions; stream just keys via lite-like subset
            raw = json.load(open(meta_p))
            meta = {
                "task_id": raw.get("task_id") or raw.get("episode_id") or os.path.basename(ep_dir),
                "frames": raw.get("frames"),
                "actions": raw.get("actions"),
                "episode_score": raw.get("episode_score"),
                "meta": raw.get("meta") or {},
            }
        else:
            meta = json.load(open(meta_p))
    frames: List[str] = []
    listed = meta.get("frames")
    if listed:
        for name in listed:
            p = name if os.path.isabs(str(name)) else os.path.join(ep_dir, os.path.basename(str(name)))
            if os.path.isfile(p):
                frames.append(p)
    if not frames:
        frames = sorted(
            os.path.join(ep_dir, n)
            for n in os.listdir(ep_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg")) and n.startswith("frame_")
        )
    actions = list(meta.get("actions") or [])
    act_path = os.path.join(ep_dir, "actions.jsonl")
    if not actions and os.path.isfile(act_path):
        with open(act_path) as f:
            for line in f:
                row = json.loads(line)
                actions.append(row.get("action_text", ""))
    actions = [wrap_action_text(a) for a in actions]
    return {
        "dir": ep_dir,
        "id": str(meta.get("task_id") or meta.get("episode_id") or os.path.basename(ep_dir)),
        "category": meta.get("category"),
        "success": meta.get("success"),
        "frames": frames,
        "actions": actions,
        "meta": meta,
    }


def score_episode_visual(
    wm,
    frames_np: Sequence[np.ndarray],
    action_texts: Sequence[str],
    *,
    device,
    nfpb: int = 3,
    max_turns: int = 24,
) -> Dict[str, Any]:
    import torch
    from curriculum.action_codec import chunk_to_mg2, parse_actions_text
    from curriculum.wm_reward_bridge import cheap_latent_from_frames

    n_turns = min(len(action_texts), max(0, len(frames_np) - 1), max_turns)
    empty = {
        "n_pairs": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "mae": 0.0,
        "latent_mse": 0.0,
    }
    if n_turns <= 0 or len(frames_np) < nfpb + 1:
        return empty

    psnrs, ssims, maes, mses = [], [], [], []
    wm.eval()
    with torch.no_grad():
        for t in range(n_turns):
            end = min(t + nfpb + 1, len(frames_np))
            start = end - (nfpb + 1)
            if start < 0:
                continue
            window = list(frames_np[start:end])
            z_t = cheap_latent_from_frames(window[:-1], nfpb=nfpb).unsqueeze(0).to(device).float()
            z_gt = cheap_latent_from_frames(window[1:], nfpb=nfpb).unsqueeze(0).to(device).float()
            acts, _ = parse_actions_text(action_texts[t])
            if acts:
                kb, ms = chunk_to_mg2(acts)
                kb = kb.mean(0, keepdim=True).to(device).float()
                ms = ms.mean(0, keepdim=True).to(device).float()
            else:
                kb = torch.zeros(1, 4, device=device)
                ms = torch.zeros(1, 2, device=device)
            z_hat = wm(z_t, kb, ms)
            mse = float(torch.mean((z_hat - z_gt) ** 2).item())
            pred_rgb = cheap_latent_to_rgb(z_hat, frame_idx=-1)
            gt_rgb = cheap_latent_to_rgb(z_gt, frame_idx=-1)
            # also compare against the actual last sandbox frame in the next window
            sandbox_next = resize_rgb(window[-1], (pred_rgb.shape[0], pred_rgb.shape[1]))
            psnrs.append(psnr(pred_rgb, sandbox_next))
            ssims.append(ssim(pred_rgb, sandbox_next))
            maes.append(mae(pred_rgb, sandbox_next))
            mses.append(mse)

    if not psnrs:
        return empty
    return {
        "n_pairs": float(len(psnrs)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "mae": float(np.mean(maes)),
        "latent_mse": float(np.mean(mses)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_std": float(np.std(ssims)),
    }


def save_preview(
    out_dir: str,
    ep_id: str,
    gt: np.ndarray,
    pred: np.ndarray,
) -> str:
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    h = max(gt.shape[0], pred.shape[0])
    w = gt.shape[1] + pred.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: gt.shape[0], : gt.shape[1]] = gt
    canvas[: pred.shape[0], gt.shape[1] :] = pred
    path = os.path.join(out_dir, f"{ep_id}_gt_left_wm_right.png")
    Image.fromarray(canvas).save(path)
    return path


def run_eval(
    *,
    traj_dir: str,
    ckpt_dir: str,
    out_dir: str,
    max_eps: int = 36,
    max_turns: int = 20,
    nfpb: int = 3,
    save_previews: int = 4,
    toy_fallback: str = "curriculum/outputs/toy_wm0.pt",
) -> Dict[str, Any]:
    import torch
    from curriculum.frozen_wm import ToyDynamics
    from PIL import Image

    out_dir = _assert_write_home(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = ToyDynamics().to(device).eval()
    loaded = None
    for p in (os.path.join(ckpt_dir, "wm_student_latest.pt"), ckpt_dir, toy_fallback):
        if p and os.path.isfile(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            state = ck.get("toy", ck.get("student", ck))
            wm.load_state_dict(state, strict=False)
            loaded = p
            break
    if loaded is None:
        print("[eval_wm_visual] warning: no WM ckpt, random init", flush=True)

    eps = _list_traj_dirs(traj_dir)[: max(1, int(max_eps))]
    rows: List[Dict[str, Any]] = []
    preview_dir = os.path.join(out_dir, "previews")
    n_prev = 0
    for ep_dir in eps:
        try:
            ep = load_episode(ep_dir)
        except Exception as e:
            print(f"[eval_wm_visual] skip {ep_dir}: {e}", flush=True)
            continue
        frames_np = []
        for p in ep["frames"]:
            try:
                frames_np.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
            except Exception:
                continue
        st = score_episode_visual(
            wm, frames_np, ep["actions"], device=device, nfpb=nfpb, max_turns=max_turns,
        )
        row = {
            "id": ep["id"],
            "category": ep.get("category"),
            "success": ep.get("success"),
            "n_frames": len(frames_np),
            **st,
        }
        rows.append(row)
        if n_prev < save_previews and len(frames_np) >= nfpb + 1:
            try:
                import torch as _t
                from curriculum.wm_reward_bridge import cheap_latent_from_frames
                from curriculum.action_codec import chunk_to_mg2, parse_actions_text

                window = frames_np[: nfpb + 1]
                z_t = cheap_latent_from_frames(window[:-1], nfpb=nfpb).unsqueeze(0).to(device).float()
                acts, _ = parse_actions_text(ep["actions"][0] if ep["actions"] else "")
                if acts:
                    kb, ms = chunk_to_mg2(acts)
                    kb = kb.mean(0, keepdim=True).to(device).float()
                    ms = ms.mean(0, keepdim=True).to(device).float()
                else:
                    kb = _t.zeros(1, 4, device=device)
                    ms = _t.zeros(1, 2, device=device)
                with _t.no_grad():
                    z_hat = wm(z_t, kb, ms)
                pred = cheap_latent_to_rgb(z_hat)
                gt = resize_rgb(window[-1], pred.shape[:2])
                save_preview(preview_dir, ep["id"], gt, pred)
                n_prev += 1
            except Exception as e:
                print(f"[eval_wm_visual] preview failed {ep['id']}: {e}", flush=True)

    def _mean(key: str) -> float:
        xs = [r[key] for r in rows if r.get("n_pairs", 0) > 0]
        return float(sum(xs) / len(xs)) if xs else 0.0

    summary = {
        "mechanism": "sandbox_gt_vs_wm_generation_visual",
        "ckpt": loaded,
        "traj_dir": traj_dir,
        "n_eps": len(rows),
        "mean_psnr": _mean("psnr"),
        "mean_ssim": _mean("ssim"),
        "mean_mae": _mean("mae"),
        "mean_latent_mse": _mean("latent_mse"),
        "note": (
            "Sandbox still decides task success. These metrics compare WM-generated "
            "next-frame (cheap-latent RGB) against sandbox next-frame."
        ),
        "episodes": rows,
    }
    out_json = os.path.join(out_dir, "eval_wm_visual.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in summary if k != "episodes"}, indent=2), flush=True)
    print(f"[eval_wm_visual] wrote {out_json}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True, help="ARES dump dir or full37 trajectories/")
    ap.add_argument("--ckpt_dir", default="curriculum/outputs/coevolve_vla_auto")
    ap.add_argument("--out", default="curriculum/outputs/eval_wm_visual")
    ap.add_argument("--max_eps", type=int, default=36)
    ap.add_argument("--max_turns", type=int, default=20)
    ap.add_argument("--nfpb", type=int, default=3)
    ap.add_argument("--previews", type=int, default=4)
    args = ap.parse_args()
    os.chdir(_ROOT)
    run_eval(
        traj_dir=args.traj_dir,
        ckpt_dir=args.ckpt_dir,
        out_dir=args.out,
        max_eps=args.max_eps,
        max_turns=args.max_turns,
        nfpb=args.nfpb,
        save_previews=args.previews,
    )


if __name__ == "__main__":
    main()
