#!/usr/bin/env python3
"""WM imagination + uncertainty gate for VLA–WM co-evolution.

Consumes ARES dumps, scores turn uncertainty with the live student WM, writes:
  - calibration_queue.jsonl  (high-uncertainty turns → sandbox re-ground candidates)
  - imagined_rollouts/*.json (latent imagination traces; NOT used to train WM)
  - reality_gap.json         (mean MSE on recent real transitions)

WM training must stay on real dumps only (coevolve_vla). Imagination is for
VLA intrinsic reward / curriculum gating / diagnostics.

Usage:
  python -m curriculum.imagine_worker \\
    --dump-dir curriculum/outputs/coevolve_vla/ares_rollouts \\
    --ckpt-dir curriculum/outputs/coevolve_vla \\
    --max-eps 32
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np


def _mg2_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _list_eps(dump_dir: str) -> List[str]:
    if not dump_dir or not os.path.isdir(dump_dir):
        return []
    eps = []
    for name in os.listdir(dump_dir):
        d = os.path.join(dump_dir, name)
        if os.path.isdir(d) and name.startswith("ep_") and os.path.isfile(os.path.join(d, "meta.json")):
            eps.append(d)
    eps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return eps


def _load_episode(ep_dir: str) -> Dict[str, Any]:
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    frames = []
    for name in meta.get("frames") or []:
        p = os.path.join(ep_dir, name)
        if os.path.isfile(p):
            frames.append(p)
    actions = []
    act_path = os.path.join(ep_dir, "actions.jsonl")
    if os.path.isfile(act_path):
        with open(act_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                actions.append(json.loads(line).get("action_text", ""))
    return {"dir": ep_dir, "meta": meta, "frames": frames, "actions": actions}


def run_once(
    dump_dir: str,
    ckpt_dir: str,
    *,
    max_eps: int = 32,
    uncert_threshold: Optional[float] = None,
    imagine_horizon: int = 4,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    import sys

    root = _mg2_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    from curriculum.wm_reward_bridge import get_scorer

    out_dir = out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    imag_dir = os.path.join(out_dir, "imagined_rollouts")
    os.makedirs(imag_dir, exist_ok=True)
    calib_path = os.path.join(out_dir, "calibration_queue.jsonl")

    if uncert_threshold is None:
        uncert_threshold = float(os.environ.get("ARES_WM_UNCERT_THRESH", "1e-4"))

    scorer = get_scorer(ckpt_dir)
    if scorer is None:
        return {"ok": False, "error": "no_scorer", "n_eps": 0}

    eps = _list_eps(dump_dir)[: max(1, int(max_eps))]
    n_calib = 0
    mses: List[float] = []
    uncerts: List[float] = []
    imagined = 0

    with open(calib_path, "a") as calib_f:
        for ep_dir in eps:
            ep = _load_episode(ep_dir)
            if len(ep["frames"]) < 4 or not ep["actions"]:
                continue
            try:
                stats = scorer.score_turns(ep["frames"], ep["actions"], all_turns=True)
            except Exception as e:
                print(f"[imagine] score failed {ep_dir}: {e}", flush=True)
                continue
            mses.append(float(stats.get("wm_mse") or 0.0))
            uncerts.append(float(stats.get("wm_uncert") or 0.0))

            turn_u = stats.get("turn_uncert") or []
            turn_m = stats.get("turn_mse") or []
            for t, u in enumerate(turn_u):
                if u is None:
                    continue
                if float(u) < uncert_threshold:
                    continue
                rec = {
                    "ts": time.time(),
                    "ep_dir": ep_dir,
                    "turn": t,
                    "uncert": float(u),
                    "mse": float(turn_m[t]) if t < len(turn_m) and turn_m[t] is not None else None,
                    "frame": ep["frames"][t] if t < len(ep["frames"]) else None,
                    "task_id": (ep["meta"].get("meta") or {}).get("task_id"),
                }
                calib_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_calib += 1

            # Short imagination trace from first scored latent (diagnostics only).
            try:
                import torch
                from curriculum.action_codec import parse_actions_text
                from PIL import Image

                t0 = 0
                frames_np = []
                for p in ep["frames"][: scorer.nfpb + 1]:
                    frames_np.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
                if len(frames_np) < scorer.nfpb:
                    continue
                z = scorer._encode_block(frames_np[: scorer.nfpb]).unsqueeze(0).to(scorer.device).float()
                traj = {"ep": os.path.basename(ep_dir), "horizon": imagine_horizon, "steps": []}
                with torch.no_grad():
                    for h in range(imagine_horizon):
                        atext = ep["actions"][min(h, len(ep["actions"]) - 1)]
                        acts, _ = parse_actions_text(atext)
                        if acts:
                            kb, ms, _ = scorer.proj.from_minestudio_chunk(acts, device=scorer.device)
                        else:
                            kb = torch.zeros(1, 4, device=scorer.device)
                            ms = torch.zeros(1, 2, device=scorer.device)
                        z_next = scorer.wm(z, kb, ms)
                        step_mse = float(torch.mean((z_next - z) ** 2).item())
                        traj["steps"].append({"h": h, "delta_mse": step_mse})
                        z = z_next
                out_p = os.path.join(imag_dir, f"{os.path.basename(ep_dir)}_H{imagine_horizon}.json")
                with open(out_p, "w") as f:
                    json.dump(traj, f)
                imagined += 1
            except Exception as e:
                print(f"[imagine] roll failed {ep_dir}: {e}", flush=True)

    gap = {
        "ts": time.time(),
        "n_eps": len(eps),
        "reality_gap_mse": float(sum(mses) / len(mses)) if mses else None,
        "mean_uncert": float(sum(uncerts) / len(uncerts)) if uncerts else None,
        "n_calib_written": n_calib,
        "n_imagined": imagined,
        "uncert_threshold": uncert_threshold,
    }
    with open(os.path.join(out_dir, "reality_gap.json"), "w") as f:
        json.dump(gap, f, indent=2)
    # Append history
    hist = os.path.join(out_dir, "reality_gap_history.jsonl")
    with open(hist, "a") as f:
        f.write(json.dumps(gap) + "\n")
    print(f"[imagine] {gap}", flush=True)
    return {"ok": True, **gap}


def write_schedule(
    out_dir: str,
    *,
    gap: Dict[str, Any],
    round_i: Optional[int] = None,
) -> str:
    """Update coevolve_schedule.json for ARES agent (λ / β / imag mix)."""
    path = os.path.join(out_dir, "coevolve_schedule.json")
    prev = {}
    if os.path.isfile(path):
        try:
            prev = json.load(open(path))
        except Exception:
            prev = {}
    r = int(round_i if round_i is not None else (prev.get("round", 0) + 1))
    # Anneal: as reality_gap drops, allow more imagination credit.
    base_lam = float(os.environ.get("ARES_WM_REWARD_LAMBDA", prev.get("lambda", 0.5)))
    base_beta = float(os.environ.get("ARES_WM_TURN_BETA", prev.get("beta", 0.5)))
    gap_mse = gap.get("reality_gap_mse")
    # Start imag_gamma low; raise toward 0.4 as round grows (capped).
    imag_gamma = min(0.4, 0.1 + 0.02 * r)
    imag_horizon = min(8, 2 + r // 2)
    if gap_mse is not None and gap_mse > 0.06:
        # WM still bad → less imagination, keep λ moderate.
        imag_gamma = min(imag_gamma, 0.1)
    sched = {
        "round": r,
        "ts": time.time(),
        "lambda": base_lam,
        "beta": base_beta,
        "imag_gamma": imag_gamma,
        "imag_horizon": imag_horizon,
        "reality_gap_mse": gap_mse,
        "mean_uncert": gap.get("mean_uncert"),
    }
    with open(path, "w") as f:
        json.dump(sched, f, indent=2)
    # Also export beta for advantage path via a sidecar env file (agent reads schedule for λ).
    with open(os.path.join(out_dir, "coevolve_schedule.env"), "w") as f:
        f.write(f"ARES_WM_TURN_BETA={base_beta}\n")
        f.write(f"ARES_WM_IMAG_GAMMA={imag_gamma}\n")
        f.write(f"ARES_WM_IMAG_HORIZON={imag_horizon}\n")
        f.write(f"ARES_WM_SCHEDULE_PATH={path}\n")
    print(f"[imagine] schedule -> {path}: {sched}", flush=True)
    return path


def maybe_sandbox_calibrate(
    out_dir: str,
    dump_dir: str,
    max_calls: int = 0,
    *,
    steps_per_call: int = 8,
) -> int:
    """Re-ground high-uncertainty turns in MineStudio; dump real transitions for WM.

    For each calibration_queue entry:
      1. reset sandbox (task_id from meta when available)
      2. replay action chunk(s) from the source episode around that turn
      3. dump POV frames + actions into dump_dir as calib_ep_* for WM training
    """
    if max_calls <= 0:
        return 0
    calib_path = os.path.join(out_dir, "calibration_queue.jsonl")
    if not os.path.isfile(calib_path):
        return 0
    try:
        from curriculum.sandbox_bridge import MineStudioSandbox, extract_pil
        from curriculum.action_codec import parse_actions_text
        from curriculum.ares_wm_bridge import dump_ares_episode
    except Exception as e:
        print(f"[imagine] sandbox deps failed: {e}", flush=True)
        return 0

    lines = open(calib_path).readlines()
    # Prefer newest high-uncert entries.
    recs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            continue
    recs.sort(key=lambda r: float(r.get("uncert") or 0.0), reverse=True)
    recs = recs[: max(1, int(max_calls))]

    done = 0
    for rec in recs:
        ep_dir = rec.get("ep_dir") or ""
        turn = int(rec.get("turn") or 0)
        if not ep_dir or not os.path.isdir(ep_dir):
            continue
        actions = []
        act_path = os.path.join(ep_dir, "actions.jsonl")
        if os.path.isfile(act_path):
            with open(act_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    actions.append(json.loads(line).get("action_text", ""))
        if not actions:
            continue
        task_id = rec.get("task_id") or (0, 0)
        if isinstance(task_id, list):
            task_id = tuple(task_id)
        if not isinstance(task_id, tuple) or len(task_id) != 2:
            task_id = (0, 0)

        sandbox = None
        try:
            sandbox = MineStudioSandbox(task_id=task_id)
            sandbox.reset()
            image_paths: List[str] = []
            action_texts: List[str] = []
            rewards: List[float] = []
            # Warm-start: take a POV after reset
            pil0 = sandbox.last_image
            tmp_root = os.path.join(out_dir, "calib_tmp", f"{int(time.time()*1000)}_{done}")
            os.makedirs(tmp_root, exist_ok=True)
            if pil0 is not None:
                p0 = os.path.join(tmp_root, "frame_0000.png")
                pil0.save(p0)
                image_paths.append(p0)

            start_t = max(0, turn)
            end_t = min(len(actions), start_t + max(1, int(steps_per_call)))
            for i, t in enumerate(range(start_t, end_t)):
                atext = actions[t]
                acts, _ = parse_actions_text(atext)
                if not acts:
                    continue
                obs = sandbox.step(acts)
                action_texts.append(atext)
                rewards.append(float(obs.get("reward", 0.0) or 0.0))
                pil = extract_pil(obs) or sandbox.last_image
                if pil is None:
                    continue
                fp = os.path.join(tmp_root, f"frame_{len(image_paths):04d}.png")
                pil.save(fp)
                image_paths.append(fp)

            if len(image_paths) >= 2 and action_texts:
                dump_ares_episode(
                    dump_dir,
                    image_paths=image_paths,
                    action_texts=action_texts,
                    rewards=rewards,
                    episode_score=float(rewards[-1] if rewards else 0.0),
                    meta={
                        "source": "sandbox_calibrate",
                        "parent_ep": ep_dir,
                        "uncert_turn": turn,
                        "uncert": rec.get("uncert"),
                        "task_id": list(task_id),
                    },
                )
                done += 1
                print(
                    f"[imagine] calibrated turn={turn} uncert={rec.get('uncert')} "
                    f"frames={len(image_paths)} -> dump",
                    flush=True,
                )
        except Exception as e:
            print(f"[imagine] sandbox calibrate failed: {e}", flush=True)
        finally:
            if sandbox is not None:
                sandbox.close()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-eps", type=int, default=32)
    ap.add_argument("--uncert-threshold", type=float, default=-1.0)
    ap.add_argument("--imagine-horizon", type=int, default=4)
    ap.add_argument("--sandbox-calibrate", type=int, default=0)
    args = ap.parse_args()
    thr = None if args.uncert_threshold < 0 else args.uncert_threshold
    out = args.out_dir or args.ckpt_dir
    gap = run_once(
        args.dump_dir,
        args.ckpt_dir,
        max_eps=args.max_eps,
        uncert_threshold=thr,
        imagine_horizon=args.imagine_horizon,
        out_dir=out,
    )
    if gap.get("ok"):
        write_schedule(out, gap=gap)
    n = maybe_sandbox_calibrate(
        out, args.dump_dir, max_calls=args.sandbox_calibrate
    )
    print(f"[imagine] sandbox_calibrate_done={n}", flush=True)


if __name__ == "__main__":
    main()
