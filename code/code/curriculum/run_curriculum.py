#!/usr/bin/env python3
"""Unified entry for WM curriculum + self-evolution.

Primary (True Auto / CLAW-style):
  python -m curriculum.run_curriculum auto_research_claw
  python -m curriculum.run_curriculum smoke_auto_research_claw

MVP (fixed Discover→Probe→Compress):
  python -m curriculum.run_curriculum auto_research_mvp
  python -m curriculum.run_curriculum smoke_auto_research_mvp

WM-only self-evolution:
  python -m curriculum.run_curriculum self_evolve_wm
  python -m curriculum.run_curriculum smoke_self_evolve_wm

WM ↔ VLA co-evolution:
  python -m curriculum.run_curriculum coevolve
  python -m curriculum.run_curriculum smoke_coevolve

Qwen GRPO + Auto (transfer vs connection):
  python -m curriculum.run_curriculum auto_vla_coevolve
  python -m curriculum.run_curriculum smoke_auto_vla_coevolve

WM visual eval (sandbox GT vs WM generation):
  python -m curriculum.run_curriculum eval_wm_visual
  python -m curriculum.run_curriculum smoke_eval_wm_visual

DiaWM probe policy:
  python -m curriculum.run_curriculum smoke_probe_policy

Training-free harness search (frozen VLA + probe + memory + sandbox):
  python -m curriculum.run_curriculum auto_harness
  python -m curriculum.run_curriculum smoke_auto_harness

Warm-up / ablations:
  stage1, pretrain_toy, stage2, stage3, stage4, smoke
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(module: str, extra: list):
    cmd = [sys.executable, "-m", module, *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command",
        choices=[
            "stage1", "pretrain_toy", "stage2", "stage3", "stage4", "smoke",
            "coevolve", "smoke_coevolve", "coevolve_vla", "smoke_coevolve_vla",
            "self_evolve_wm", "smoke_self_evolve_wm",
            "auto_research_mvp", "smoke_auto_research_mvp",
            "auto_research_claw", "smoke_auto_research_claw",
            "auto_vla_coevolve", "smoke_auto_vla_coevolve",
            "eval_wm_visual", "smoke_eval_wm_visual",
            "smoke_probe_policy",
            "auto_harness", "smoke_auto_harness",
            "test_auto_harness", "ping_minestudio", "ping_gemini",
            "harness_vla", "smoke_harness_vla",
            "eval_auto_research",
        ],
    )
    ap.add_argument("--allow_missing", action="store_true")
    ap.add_argument("--config", default=None)
    args, unknown = ap.parse_known_args()
    os.chdir(_ROOT)

    if args.command == "stage1":
        extra = ["--out", "curriculum/outputs/stage1_bundle.json"]
        if args.allow_missing:
            extra.append("--allow_missing")
        _run("curriculum.stage1_bundle", extra + unknown)

    elif args.command == "pretrain_toy":
        _run("curriculum.pretrain_toy_wm", [
            "--out", "curriculum/outputs/toy_wm0.pt",
            "--steps", "200",
        ] + unknown)

    elif args.command == "stage2":
        cfg = args.config or "curriculum/configs/stage2_frozen_wm_vla.yaml"
        from omegaconf import OmegaConf
        c = OmegaConf.load(cfg)
        if os.path.exists("curriculum/outputs/toy_wm0.pt"):
            c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
            tmp = "curriculum/outputs/_stage2_runtime.yaml"
            OmegaConf.save(c, tmp)
            cfg = tmp
        _run("curriculum.stage2_vla_rl", ["--config", cfg] + unknown)

    elif args.command == "stage3":
        cfg = args.config or "curriculum/configs/stage3_wm_adapt.yaml"
        from omegaconf import OmegaConf
        c = OmegaConf.load(cfg)
        if os.path.exists("curriculum/outputs/toy_wm0.pt") and not c.get("toy_wm_ckpt"):
            c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        pol = "curriculum/outputs/stage2/policy_latest.pt"
        if os.path.exists(pol):
            c.policy_ckpt = pol
        tmp = "curriculum/outputs/_stage3_runtime.yaml"
        OmegaConf.save(c, tmp)
        _run("curriculum.stage3_wm_adapt", ["--config", tmp] + unknown)

    elif args.command == "stage4":
        cfg = args.config or "curriculum/configs/stage4_closed_loop.yaml"
        _run("curriculum.stage4_closed_loop", ["--config", cfg] + unknown)

    elif args.command == "self_evolve_wm":
        cfg = args.config or "curriculum/configs/self_evolve_wm.yaml"
        from omegaconf import OmegaConf
        c = OmegaConf.load(cfg)
        if os.path.exists("curriculum/outputs/toy_wm0.pt") and not c.get("toy_wm_ckpt"):
            c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        tmp = "curriculum/outputs/_self_evolve_wm_runtime.yaml"
        OmegaConf.save(c, tmp)
        _run("curriculum.self_evolve_wm", ["--config", tmp] + unknown)

    elif args.command == "smoke_self_evolve_wm":
        from omegaconf import OmegaConf
        if not os.path.exists("curriculum/outputs/toy_wm0.pt"):
            _run("curriculum.pretrain_toy_wm", [
                "--out", "curriculum/outputs/toy_wm0.pt",
                "--data_root", "data/mc_vpt_train",
                "--steps", "80",
                "--max_clips", "40",
            ])
        c = OmegaConf.load("curriculum/configs/self_evolve_wm.yaml")
        c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        c.data_root = "data/mc_vpt_train"
        c.max_clips = 20
        c.use_sandbox = True
        c.sandbox_backend = "stub"  # no network for smoke
        c.latent_encoder = "cheap"
        c.max_steps = 12
        c.batch_size = 16
        c.probe_n = 8
        c.sandbox_max_turns = 4
        c.collect_every = 1
        c.log_interval = 4
        c.save_interval = 6
        c.logdir = "curriculum/outputs/smoke_self_evolve_wm"
        OmegaConf.save(c, "curriculum/outputs/_smoke_self_evolve_wm.yaml")
        _run("curriculum.self_evolve_wm", [
            "--config", "curriculum/outputs/_smoke_self_evolve_wm.yaml",
        ])
        print("\n[smoke_self_evolve_wm] OK — sandbox(stub) → probe → memory → WM update.")

    elif args.command == "auto_research_mvp":
        cfg = args.config or "curriculum/configs/auto_research_gaming_mvp.yaml"
        from omegaconf import OmegaConf
        c = OmegaConf.load(cfg)
        if os.path.exists("curriculum/outputs/toy_wm0.pt"):
            c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        tmp = "curriculum/outputs/_auto_research_mvp_runtime.yaml"
        OmegaConf.save(c, tmp)
        _run("curriculum.auto_research_mvp", ["--config", tmp] + unknown)

    elif args.command == "smoke_auto_research_mvp":
        from omegaconf import OmegaConf
        if not os.path.exists("curriculum/outputs/toy_wm0.pt"):
            _run("curriculum.pretrain_toy_wm", [
                "--out", "curriculum/outputs/toy_wm0.pt",
                "--data_root", "data/mc_vpt_train",
                "--steps", "80",
                "--max_clips", "40",
            ])
        c = OmegaConf.load("curriculum/configs/auto_research_gaming_mvp.yaml")
        c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        c.sandbox_backend = "stub"
        c.max_steps = 10
        c.max_sandbox_episodes = 10
        c.max_wm_updates = 10
        c.batch_size = 16
        c.probe_n = 8
        c.sandbox_max_turns = 4
        c.log_interval = 2
        c.save_interval = 5
        c.logdir = "curriculum/outputs/smoke_auto_research_mvp"
        OmegaConf.save(c, "curriculum/outputs/_smoke_auto_research_mvp.yaml")
        _run("curriculum.auto_research_mvp", [
            "--config", "curriculum/outputs/_smoke_auto_research_mvp.yaml",
        ])
        print("\n[smoke_auto_research_mvp] OK — frozen contract + evidence + holdout.")

    elif args.command == "auto_research_claw":
        cfg = args.config or "curriculum/configs/auto_research_gaming_claw.yaml"
        from omegaconf import OmegaConf
        c = OmegaConf.load(cfg)
        if os.path.exists("curriculum/outputs/toy_wm0.pt"):
            c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        tmp = "curriculum/outputs/_auto_research_claw_runtime.yaml"
        OmegaConf.save(c, tmp)
        _run("curriculum.auto_research_claw", ["--config", tmp] + unknown)

    elif args.command == "smoke_auto_research_claw":
        from omegaconf import OmegaConf
        if not os.path.exists("curriculum/outputs/toy_wm0.pt"):
            _run("curriculum.pretrain_toy_wm", [
                "--out", "curriculum/outputs/toy_wm0.pt",
                "--data_root", "data/mc_vpt_train",
                "--steps", "80",
                "--max_clips", "40",
            ])
        c = OmegaConf.load("curriculum/configs/auto_research_gaming_claw.yaml")
        c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        c.sandbox_backend = "stub"
        c.max_steps = 12
        c.max_sandbox_episodes = 12
        c.max_wm_updates = 12
        c.batch_size = 16
        c.probe_n = 8
        c.hyp_topk = 10
        c.sandbox_max_turns = 4
        c.reverify_every = 3
        c.mastered_min_verify = 1
        c.mastered_min_conf = 0.35
        c.log_interval = 2
        c.save_interval = 6
        c.logdir = "curriculum/outputs/smoke_auto_research_claw"
        OmegaConf.save(c, "curriculum/outputs/_smoke_auto_research_claw.yaml")
        _run("curriculum.auto_research_claw", [
            "--config", "curriculum/outputs/_smoke_auto_research_claw.yaml",
        ])
        print("\n[smoke_auto_research_claw] OK — WorthLearn Learn|Skip|Re-verify scheduler.")

    elif args.command == "auto_vla_coevolve":
        cfg = args.config or "curriculum/configs/auto_vla_coevolve.yaml"
        from omegaconf import OmegaConf
        c = OmegaConf.load(cfg)
        if os.path.exists("curriculum/outputs/toy_wm0.pt"):
            c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        tmp = "curriculum/outputs/_auto_vla_coevolve_runtime.yaml"
        OmegaConf.save(c, tmp)
        _run("curriculum.auto_vla_coevolve", ["--config", tmp] + unknown)

    elif args.command == "smoke_auto_vla_coevolve":
        from omegaconf import OmegaConf
        from curriculum.auto_vla_coevolve import synth_dump_episodes
        if not os.path.exists("curriculum/outputs/toy_wm0.pt"):
            _run("curriculum.pretrain_toy_wm", [
                "--out", "curriculum/outputs/toy_wm0.pt",
                "--data_root", "data/mc_vpt_train",
                "--steps", "80",
                "--max_clips", "40",
            ])
        dump = "curriculum/outputs/smoke_auto_vla_coevolve/ares_rollouts"
        synth_dump_episodes(dump, n_eps=4, n_frames=10, seed=1)
        c = OmegaConf.load("curriculum/configs/auto_vla_coevolve.yaml")
        c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        c.dump_dir = dump
        c.logdir = "curriculum/outputs/smoke_auto_vla_coevolve"
        c.max_steps = 6
        c.max_eps_per_step = 4
        c.probe_n = 8
        c.hyp_topk = 8
        c.wait_dump = False
        c.poll_s = 0.1
        OmegaConf.save(c, "curriculum/outputs/_smoke_auto_vla_coevolve.yaml")
        _run("curriculum.auto_vla_coevolve", [
            "--config", "curriculum/outputs/_smoke_auto_vla_coevolve.yaml",
        ])
        print("\n[smoke_auto_vla_coevolve] OK — Qwen GRPO dump → WorthLearn → schedule.json transfer λ.")

    elif args.command == "eval_wm_visual":
        extra = [
            "--ckpt_dir", "curriculum/outputs/coevolve_vla_auto",
            "--out", "curriculum/outputs/eval_wm_visual",
        ]
        if any(a == "--traj_dir" for a in unknown):
            extra = []
        _run("curriculum.eval_wm_visual", extra + unknown)

    elif args.command == "smoke_eval_wm_visual":
        from curriculum.auto_vla_coevolve import synth_dump_episodes
        dump = "curriculum/outputs/smoke_eval_wm_visual/ares_rollouts"
        synth_dump_episodes(dump, n_eps=3, n_frames=10, seed=2)
        _run("curriculum.eval_wm_visual", [
            "--traj_dir", dump,
            "--ckpt_dir", "curriculum/outputs/coevolve_vla_auto",
            "--out", "curriculum/outputs/smoke_eval_wm_visual",
            "--max_eps", "3",
            "--max_turns", "8",
            "--previews", "2",
        ])
        print("\n[smoke_eval_wm_visual] OK — sandbox GT vs WM RGB (PSNR/SSIM/MAE).")

    elif args.command == "smoke_probe_policy":
        import json
        import torch
        from curriculum.frozen_wm import ToyDynamics
        from curriculum.knowledge_memory import WorldKnowledgeMemory
        from curriculum.probe_policy import score_probe_episode, probe_focus_actions
        from curriculum.replay_buffer import Transition
        wm = ToyDynamics()
        mem = WorldKnowledgeMemory(min_confidence=0.1)
        batch = []
        for i in range(6):
            z = torch.randn(16, 3, 8, 8)
            kb = torch.zeros(9, 4)
            kb[:, i % 4] = 1.0
            ms = torch.zeros(9, 2)
            z2 = z + 0.05 * torch.randn_like(z)
            tr = Transition(latent_t=z, keyboard=kb, mouse=ms, latent_tp1=z2, meta={"action": ["forward","back","left","right"][i%4]})
            mem.propose(z, kb, ms, z2, probe_scores={"action_causality": 0.6, "counterfactual": 0.55}, confidence=0.6, meta=tr.meta)
            batch.append(tr)
        st = score_probe_episode(wm, batch, mem, device=torch.device("cpu"), uncert=0.4)
        assert "r_probe" in st and "gap" in st and len(st["gap"]) == 8
        focus = probe_focus_actions(mem, k=3)
        print(json.dumps({"r_probe": round(st["r_probe"], 4), "gap_norm": round(st["gap_norm"], 4),
                          "ranking": round(st["ranking"], 4), "focus": focus}, indent=2))
        print("\n[smoke_probe_policy] OK — DiaWM r_probe + gap vector g.")

    elif args.command == "auto_harness":
        cfg = args.config or "curriculum/configs/auto_harness.yaml"
        _run("curriculum.auto_harness_search", ["--config", cfg] + unknown)

    elif args.command == "smoke_auto_harness":
        _run("curriculum.auto_harness_search", [
            "--smoke",
            "--logdir", "curriculum/outputs/smoke_auto_harness",
            "--rounds", "8",
            "--seed", "0",
        ] + unknown)

    elif args.command == "test_auto_harness":
        _run("curriculum.test_auto_harness", unknown)

    elif args.command == "ping_minestudio":
        _run("curriculum.auto_harness_search", ["--ping-minestudio"] + unknown)

    elif args.command == "ping_gemini":
        _run("curriculum.auto_harness_search", ["--ping-llm"] + unknown)

    elif args.command == "harness_vla":
        cfg = args.config or "curriculum/configs/harness_vla_minestudio.yaml"
        _run("curriculum.harness_vla_agent", ["--config", cfg] + unknown)

    elif args.command == "smoke_harness_vla":
        _run("curriculum.harness_vla_agent", ["--smoke"] + unknown)

    elif args.command == "eval_auto_research":
        run_dir = "curriculum/outputs/auto_research_gaming_full"
        # allow --run_dir via unknown or --config abused; prefer explicit unknown passthrough
        extra = ["--run_dir", run_dir]
        # if user passed --run_dir in unknown, let module parse it: drop default
        if any(a == "--run_dir" for a in unknown):
            extra = []
        _run("curriculum.eval_auto_research", extra + unknown)

    elif args.command == "coevolve":
        cfg = args.config or "curriculum/configs/coevolve.yaml"
        from omegaconf import OmegaConf
        c = OmegaConf.load(cfg)
        if os.path.exists("curriculum/outputs/toy_wm0.pt"):
            c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        tmp = "curriculum/outputs/_coevolve_runtime.yaml"
        OmegaConf.save(c, tmp)
        _run("curriculum.coevolve", ["--config", tmp] + unknown)

    elif args.command == "coevolve_vla":
        cfg = args.config or "curriculum/configs/coevolve_vla.yaml"
        _run("curriculum.coevolve_vla", ["--config", cfg] + unknown)

    elif args.command == "smoke_coevolve_vla":
        from omegaconf import OmegaConf
        if not os.path.exists("curriculum/outputs/toy_wm0.pt"):
            _run("curriculum.pretrain_toy_wm", [
                "--out", "curriculum/outputs/toy_wm0.pt", "--steps", "50",
            ])
        c = OmegaConf.load("curriculum/configs/coevolve_vla_stub.yaml")
        c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        c.max_steps = 10
        c.logdir = "curriculum/outputs/smoke_coevolve_vla"
        OmegaConf.save(c, "curriculum/outputs/_smoke_coevolve_vla.yaml")
        _run("curriculum.coevolve_vla", ["--config", "curriculum/outputs/_smoke_coevolve_vla.yaml"])
        print("\n[smoke_coevolve_vla] OK — stub VLA+WM loop (no sandbox/8B).")

    elif args.command == "smoke_coevolve":
        from omegaconf import OmegaConf
        _run("curriculum.stage1_bundle", [
            "--out", "curriculum/outputs/stage1_bundle.json", "--allow_missing",
        ])
        _run("curriculum.pretrain_toy_wm", [
            "--out", "curriculum/outputs/toy_wm0.pt", "--steps", "100",
        ])
        c = OmegaConf.load("curriculum/configs/coevolve.yaml")
        c.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        c.max_steps = 40
        c.log_interval = 5
        c.save_interval = 20
        c.logdir = "curriculum/outputs/smoke_coevolve"
        OmegaConf.save(c, "curriculum/outputs/_smoke_coevolve.yaml")
        _run("curriculum.coevolve", ["--config", "curriculum/outputs/_smoke_coevolve.yaml"])
        print("\n[smoke_coevolve] OK — CoMAP-style co-evolution completed (toy).")

    elif args.command == "smoke":
        # legacy alternating stages (ablation)
        _run("curriculum.stage1_bundle", [
            "--out", "curriculum/outputs/stage1_bundle.json", "--allow_missing",
        ])
        _run("curriculum.pretrain_toy_wm", [
            "--out", "curriculum/outputs/toy_wm0.pt", "--steps", "100",
        ])
        from omegaconf import OmegaConf
        c2 = OmegaConf.load("curriculum/configs/stage2_frozen_wm_vla.yaml")
        c2.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        c2.max_steps = 20
        c2.log_interval = 5
        c2.save_interval = 20
        c2.logdir = "curriculum/outputs/smoke_stage2"
        OmegaConf.save(c2, "curriculum/outputs/_smoke_stage2.yaml")
        _run("curriculum.stage2_vla_rl", ["--config", "curriculum/outputs/_smoke_stage2.yaml"])

        c3 = OmegaConf.load("curriculum/configs/stage3_wm_adapt.yaml")
        c3.toy_wm_ckpt = "curriculum/outputs/toy_wm0.pt"
        c3.policy_ckpt = "curriculum/outputs/smoke_stage2/policy_latest.pt"
        c3.max_steps = 20
        c3.logdir = "curriculum/outputs/smoke_stage3"
        OmegaConf.save(c3, "curriculum/outputs/_smoke_stage3.yaml")
        _run("curriculum.stage3_wm_adapt", ["--config", "curriculum/outputs/_smoke_stage3.yaml"])

        c4 = OmegaConf.load("curriculum/configs/stage4_closed_loop.yaml")
        c4.num_rounds = 2
        c4.stage2.max_steps = 15
        c4.stage3.max_steps = 15
        c4.logdir = "curriculum/outputs/smoke_stage4"
        import shutil
        os.makedirs(c4.logdir, exist_ok=True)
        shutil.copy2("curriculum/outputs/toy_wm0.pt", os.path.join(c4.logdir, "shared_wm.pt"))
        OmegaConf.save(c4, "curriculum/outputs/_smoke_stage4.yaml")
        _run("curriculum.stage4_closed_loop", ["--config", "curriculum/outputs/_smoke_stage4.yaml"])
        print("\n[smoke] OK — legacy alternating loop completed.")


if __name__ == "__main__":
    main()
