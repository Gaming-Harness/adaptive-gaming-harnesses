# Async VLA–World-Model Co-Evolution via Consistency-Shaped GRPO

**Working title / shorthand:** CoEvolve-VLA (or ACE: Async Consistency Co-Evolution)  
**Venue target:** workshop → conference (ICLR / NeurIPS / CoRL / IROS)  
**Scope:** Minecraft / MineStudio / OpenHA. Qwen3-VL GRPO + project_root latent WM.

**This RP is independent of DiaWM** (`docs/RP_DiaWM.md`: probes, memory, revoke).  
Here the claim is only: **the agent and the world model improve each other asynchronously.**

**Codebase (anonymous):**  
`project_root/curriculum/{coevolve_vla,wm_reward_bridge,action_codec}.py`  
`ares/tasks/.../minestudio_qwen3_vl_8b/{config_coevolve_wm,run_coevolve_wm,submit_coevolve_wm.hope}`

---

## Abstract

Vision-language-action (VLA) agents and video / latent world models (WMs) are usually trained **in sequence**: freeze a WM on offline data, then RL the policy—or freeze the VLA and only fit \(F_\phi\). Both break under non-stationarity: \(p_\pi(s,a)\) drifts away from \(p_{\mathrm{data}}\).

We propose **async co-evolution** for Minecraft VLAs. A Qwen3-VL policy is optimized with GRPO on MineStudio. Completed episodes are dumped to a **WM sidecar** that updates an action-conditioned latent dynamics \(F_\phi\). The same \(F_\phi\) (loaded by checkpoint mtime) **reshapes GRPO returns** by latent consistency under a **hard action codec** (VLA text → MineStudio → pretrained_wm keyboard/camera). Imagined pixels never enter the VLM GRPO buffer.

Empirically we target OpenHA-32: (i) hold-out \(L_{\mathrm{dyn}}\) decreases on on-policy dumps; (ii) \(\lambda>0\) consistency-shaped GRPO vs \(\lambda=0\) env-only; (iii) ablations of codec / open-loop action heads. The action codec is **glue**, not the novelty.

---

## 1. Introduction

**Why co-evolve.**  
A capable gaming agent needs a WM that tracks the current policy, and a policy that can use a better WM. Fixed-WM RL fails because \(p_\pi \neq p_{\mathrm{data}}\) (CoMAP-style critique). Alternating “stage1 WM → stage2 freeze → VLA” is an ablation, not the algorithm.

**Why async, not one joint job.**  
Qwen3-VL GRPO lives on the cluster (cluster job system / trainer). project_root \(F_\phi\) is cheaper and can run as a **sidecar**. Coupling them in one process is operationally brittle (sandbox quota, GPU mix, restart races). File-based dump ↔ ckpt handshake is the production interface.

**Why VLM-safe consistency.**  
Qwen GRPO consumes images + text. There is **no image decoder** from pretrained_wm latents back into the VLM context. Therefore imagined \(\hat z\) **cannot** be inserted into the GRPO buffer. The only legal use of \(F_\phi\) inside RL is a **scalar consistency bonus** on real frames.

**Why a hard codec.**  
VLA emits free-form `<actions> keyPress(w); ... </actions>`; pretrained_wm expects \((\mathrm{kb}[4],\mathrm{mouse}[2])\). Learnable ActionProjection and open-loop \(\pi(z)\to a\) (Cascaded WAM) were tried and **poisoned** \(F_\phi\) when \(\pi\) was bad. The stable recipe uses a deterministic hard map only.

**Contributions.**

1. **Async dump–sidecar–reward loop** for a production VLA (Qwen3-VL GRPO) and a visual/latent WM (pretrained_wm ToyDynamics / student \(F_\phi\)).
2. **VLM-safe consistency shaping:** \(R = R_{\mathrm{env}} + \lambda \cdot u(\|F_\phi(z,a)-z'\|^2)\) on real transitions; no imagined frames in GRPO.
3. **Hard action codec** as the shared interface for WM training and scoring; negative result on learnable proj / \(L_{\mathrm{open}}\).
4. **OpenHA protocol:** \(\lambda=0\) A/B, `eval_wm` before/after sidecar rounds, codec coverage.

---

## 2. Related Work

| Line | Relation |
|------|----------|
| Dreamer / video WMs / prior_world_model | Backbone for \(F_\phi\); we do not claim a new video generator |
| VLA RL (OpenHA, RT-2-style, GRPO on VLMs) | We add an evolving WM into the **return**, not into the token stream |
| VLAW / CoMAP | Closest in spirit; CoMAP is textual LLM agents. We do **visual** WM + **VLA** + sandbox gaming, **async** |
| Joint actor–critic world models | Usually one process / one latent. We cannot put \(\hat z\) into Qwen |

DiaWM (probes + memory) is **out of scope** for this paper.

---

## 3. Method

### 3.1 Setup

Policy \(\pi_\theta\): Qwen3-VL, GRPO on MineStudio OpenHA tasks.  
Dynamics \(F_\phi:\mathcal{Z}\times\mathcal{A}\to\mathcal{Z}\). Latent \(z\) is a **shared cheap encoder** (or VAE) **identical** in sidecar ingest and ARES scorer.

Action:
\[
a_t=\mathrm{HardCodec}(\mathrm{text}_t)\in\mathbb{R}^4\times\mathbb{R}^2.
\]

### 3.2 Architecture

```text
┌────────────── ARES cluster scheduler (GRPO) ──────────────┐
│  π_θ  +  MineStudio  +  r_env                │
│  episode end → dump POV + action_text        │
└─────────────────────┬────────────────────────┘
                      │  ares_rollouts/ep_*
                      ▼
┌────────────── WM sidecar ────────────────────┐
│  encode z, hard-codec a, update F_φ (L_dyn)  │
│  save wm_student_latest.pt (mtime)           │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
         ARES scorer reloads ckpt
         R ← r_env + λ · consistency(F_φ)
```

Not one cluster scheduler job: GRPO on cluster; WM on local/sidecar GPU.

### 3.3 Hard codec (interface)

\[
\mathrm{text}\xrightarrow{\mathrm{parse}}\mathrm{MineStudio\ dict}\xrightarrow{\mathrm{sandbox\_to\_mg2}}(\mathrm{kb},\mathrm{mouse}).
\]
MVP: \(\{w,s,a,d\}\) + camera. Attack / click / use often map to **zeros** (limitation; optional enrichment later). Same codec in \(F_\phi\) training and L2 scoring.

**Disabled by default:** `train_action_proj`, Cascaded ActionHead \(\pi(z)\to a\), \(\lambda_{\mathrm{open}}>0\) (open-loop \(F(z,\pi(z))\) corrupted dynamics).

### 3.4 WM update

From dumps, minimize
\[
\mathcal{L}_{\mathrm{WM}}=\mathcal{L}_{\mathrm{dyn}}+\lambda_{\mathrm{sd}}\mathcal{L}_{\mathrm{sd}},
\qquad \mathcal{L}_{\mathrm{dyn}}=\|F_\phi(z_t,a_t)-z_{t+1}\|^2.
\]
Short rounds: \(\mathrm{wm\_epochs}\approx 3\) (~1k steps), `eval_wm` before/after. EMA teacher optional (self-distill).

### 3.5 Consistency-shaped GRPO

For real pairs \((z_t,a_t,z_{t+1})\):
\[
\mathrm{mse}=\|F_\phi(z_t,a_t)-z_{t+1}\|^2,\qquad
u=\mathrm{unit}(\mathrm{mse};\,\mathrm{mse_{ref}},\tau)\in(-1,1),
\]
\[
R = R_{\mathrm{env}}+\lambda\,u.
\]
Bring-up: \(\lambda\approx 0.5\), episode-level bonus, **no** per-turn credit, imagination \(\gamma=0\). Multi-step imag vs **real** future frames can be added later; still no fake images in GRPO.

### 3.6 What this is not

- Not a full WAM: pretrained_wm does not generate VLA actions coupled with \(F\).  
- Not diagnostic memory / intervention probes (DiaWM).  
- Not joint backprop through Qwen and \(F_\phi\).

---

## 4. Experiments

### Research questions

- **RQ1 (WM):** On-policy dumps + sidecar reduce hold-out \(L_{\mathrm{dyn}}\) vs frozen \(F_\phi\)?  
  *Preliminary:* \(0.386\to 0.313\) (\(\Delta-0.073\)) on one hard-codec round.
- **RQ2 (Policy):** \(\lambda>0\) vs \(\lambda=0\) GRPO on OpenHA-32 success / return?
- **RQ3 (Interface):** Hard codec vs learnable proj vs \(\lambda_{\mathrm{open}}>0\) (expected: latter two hurt \(F_\phi\)).
- **RQ4 (Ablation):** λ, wm_epochs, cheap-latent vs VAE, episode vs turn credit.

### Baselines

| Name | Description |
|------|-------------|
| GRPO λ=0 | Env reward only (same dump/codec unused for R) |
| Frozen WM + λ | Sidecar off; fixed \(F_\phi\) scorer |
| CoEvolve λ>0 | Full async loop (**ours**) |
| Staged | Stage1 WM offline → freeze → GRPO |
| Learnable proj / \(L_{\mathrm{open}}\) | Negative / stress ablations |

### Metrics

OpenHA task success (ID); hold-out \(L_{\mathrm{dyn}}\); codec non-zero rate; training stability (ckpt iteration, empty attempts).

### Stable recipe

```text
advantage = grpo
λ ≈ 0.5, turn_credit = 0, imag_γ = 0
train_action_proj = false, λ_open = 0
wm_epochs ≈ 3 + eval_wm
one cluster scheduler job only (sandbox CPU quota)
```

---

## 5. Status & risks (honest)

| Item | Status |
|------|--------|
| Dump → sidecar → `wm_student_latest.pt` | Implemented |
| Hard codec in F and scorer | Implemented |
| \(L_{\mathrm{dyn}}\) drop | Observed once |
| λ=0 A/B on VLA success | **Not yet conclusive** (cluster scheduler INITIALIZED / quota / failover) |
| Codec drop (attack/click) | Known; may zero many actions |

Systems: sandbox ~4000-core quota; BeeGFS `torch.save` tmp fail kills sidecar; keep **one** GRPO job.

---

## 6. Timeline

| Month | Milestone |
|-------|-----------|
| 1 | Stabilize cluster scheduler + sidecar I/O; finish λ=0 A/B |
| 2 | Enrich codec (attack/use); λ / epochs sweep; report \(L_{\mathrm{dyn}}\) + success |
| 3 | Frozen-WM vs coevolve; staged baseline; OOD OpenHA split |
| 4 | Write-up; release dump format + sidecar |

---

## 7. FAQ

**Q: Is the hard codec the contribution?**  
A: No. Necessary interface. Contribution is **async consistency co-evolution** under VLM constraints.

**Q: Why not put imagination into GRPO?**  
A: No decoder from \(\hat z\) to Qwen images; doing so would be a different (heavier) system.

**Q: Why not one joint trainer?**  
A: Cluster GRPO vs local WM; async files are the robust production split. Algorithmically still co-evolution: \(\phi\) tracks \(\pi\), \(\pi\) is scored by \(\phi\).

**Q: Relation to DiaWM?**  
A: Orthogonal next paper. This RP must stand with λ A/B + \(L_{\mathrm{dyn}}\) alone.

---

## 8. Code map

| Piece | Path |
|-------|------|
| Sidecar update | `curriculum/coevolve_vla.py --update-from-dir` |
| Hard codec | `curriculum/action_codec.py` |
| GRPO consistency | `curriculum/wm_reward_bridge.py` + agent `_apply_wm_reward` |
| cluster scheduler | `submit_coevolve_wm.hope`, `run_coevolve_wm.sh`, `config_coevolve_wm.py` |
| Launch | `run_wm_sidecar.sh`, `run_coevolve_full.sh` |

---

## One-sentence pitch

> **The VLA dumps teach the world model; the world model’s consistency teaches the VLA—asynchronously, and without feeding fake pixels into a VLM.**
