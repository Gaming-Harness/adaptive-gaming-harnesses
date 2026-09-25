#!/usr/bin/env python3
"""Stage 1: package & freeze a usable WM imagination substrate.

Does NOT retrain F from scratch — verifies existing pretrained_wm + close-loop checkpoints
and writes a frozen bundle manifest for Stages 2–4.

Usage:
  python -m curriculum.stage1_bundle --out curriculum/outputs/stage1_bundle.json
  python -m curriculum.stage1_bundle --verify curriculum/outputs/stage1_bundle.json
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.frozen_wm import Stage1Bundle, _mg2_root


DEFAULT_BUNDLE = Stage1Bundle(
    generator_ckpt="pretrained_wm/base_distilled_model/base_distill.safetensors",
    config_path="configs/inference_yaml/inference_universal.yaml",
    pretrained_model_path="pretrained_wm",
    predictor_ckpt="close-loop/outputs/long45/checkpoints/predictor.pt",
    hamiltonian_ae_ckpt="close-loop/outputs/long45/checkpoints/hamiltonian_ae.pt",
    phase_world_ckpt="close-loop/outputs/long45/checkpoints/phase_world.pt",
    oracle_reground_ckpt="close-loop/outputs/long45/checkpoints/oracle_reground.pt",
    notes=(
        "Stage-1 frozen teacher for curriculum. "
        "F = base_distill (+ optional F-phase finetune). "
        "Long-horizon stack: DriftPredictor + Hamiltonian AE / phase_world / oracle re-ground. "
        "Do not update these weights during Stage-2 VLA RL."
    ),
)


def build_default(out_path: str, allow_missing: bool = False) -> Stage1Bundle:
    bundle = DEFAULT_BUNDLE
    missing = bundle.verify(_mg2_root())
    if missing and not allow_missing:
        print("[stage1] missing files:")
        for m in missing:
            print("  -", m)
        print("[stage1] pass --allow_missing to write manifest anyway "
              "(toy backend does not need full WM).")
        sys.exit(1)
    if missing:
        print("[stage1] WARNING missing (writing anyway):")
        for m in missing:
            print("  -", m)
    bundle.save(out_path)
    print(f"[stage1] wrote frozen bundle -> {out_path}")
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="curriculum/outputs/stage1_bundle.json")
    ap.add_argument("--verify", default=None, help="verify an existing bundle json")
    ap.add_argument("--allow_missing", action="store_true")
    ap.add_argument(
        "--generator_ckpt",
        default=None,
        help="override generator checkpoint (e.g. F-phase finetune)",
    )
    args = ap.parse_args()
    os.chdir(_ROOT)

    if args.verify:
        b = Stage1Bundle.load(args.verify)
        missing = b.verify(_ROOT)
        if missing:
            print("MISSING:")
            for m in missing:
                print(" ", m)
            sys.exit(1)
        print(f"[stage1] OK: {args.verify}")
        print(json_dumps(b.to_dict()))
        return

    if args.generator_ckpt:
        DEFAULT_BUNDLE.generator_ckpt = args.generator_ckpt
    build_default(args.out, allow_missing=args.allow_missing)


def json_dumps(d):
    import json
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    main()
