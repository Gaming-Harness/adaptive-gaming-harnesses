# WM Self-Evolution + VLA Co-Evolution (project_root)

## Reliable Auto Research MVP (Gaming WM)

```bash
# smoke (stub sandbox, frozen contract)
python -m curriculum.run_curriculum smoke_auto_research_mvp

# full MVP config (set sandbox_backend: minestudio|llm_api for live)
python -m curriculum.run_curriculum auto_research_mvp
```

Artifacts under `curriculum/outputs/auto_research_gaming_mvp/`:
- `research_contract.json` — frozen budget/seeds/probes/fingerprint
- `evidence_graph.json` — hypothesis / patch / probe / boundary / revoke
- `mvp_report.json` — verdict

Mechanism under test (not AutoDL): `causal_intervene_selective_update`.

---

## Adaptive probing around a frozen agent

The original probing policy is `selection_strategy: coverage`: warm-up,
failure, stall, or periodic triggers select an uncovered action (otherwise a
random action). It writes experience, but probe outcomes do not change the
probing policy itself.

Set `probe_selection_strategy: ucb` to enable the training-free feedback loop:

```text
(task family × trigger) → select probe → sandbox transition
                        → immediate utility (progress/reward/novelty/cost/fail)
                        → delayed task-success credit → contextual UCB table
                                                    ↘ global transfer prior
```

The frozen VLA is never updated. The learned JSON table can be persisted with
`probe_bandit_state_path`; every episode exposes `meta.probe_events` and
`meta.probe_bandit` for auditing. Paired ACCEPT/REVERT evaluation resets the
table per seed, so candidates cannot benefit from evaluation-order leakage.

```bash
python -m curriculum.auto_harness_search \
  --config curriculum/configs/auto_harness_adaptive_probe.yaml
```

Key implementation: `contextual_probe_bandit.py`, `harness_runtime.py`, and the
`adaptive_ucb` mutation in `auto_harness_search.py`.

---

## WM-only self-evolution (primary for current RP)

**RP claim:** Automatically discover capabilities → probe whether they work →
compress only verified ones into experience memory.

```text
Sandbox interact
   → ① Discover  candidate capabilities (novelty / uncertainty / under-explore)
   → ② Probe     intervention tests (causal / counterfactual / transfer / horizon)
   → ③ Compress  verified → experience memory M ; discard failures
   → Selective update of F_φ on failures + verified priors
```

Raw trajectories are **not** the memory. Memory stores compressed verified
`(condition, action, effect, confidence)` items.

```bash
# warm-up toy F_φ0 (if needed)
python -m curriculum.run_curriculum pretrain_toy

# smoke (stub sandbox, no network)
python -m curriculum.run_curriculum smoke_self_evolve_wm

# ★ MineStudio live sandbox (WM-only, no VLA)
# token: set MINESTUDIO_TOKEN
python -m curriculum.run_curriculum self_evolve_wm \
  --config curriculum/configs/self_evolve_wm.yaml

# True counterfactual fork (LLMAPI MineRL; needs LLM_SANDBOX_*)
# edit yaml: sandbox_backend: llm_api
```

Modules: `capability_discovery.py`, `sandbox_experience.py`, `probes.py`,
`knowledge_memory.py`, `self_evolve_wm.py`.

`sandbox_backend`:
| value | meaning |
|-------|---------|
| `stub` | offline unit test |
| `minestudio` | online MineStudio (same gateway as coevolve_vla) |
| `llm_api` | seed-reset MineRL → true A/B fork interventions |

---

## WM ↔ VLA co-evolution (downstream)

**Claim:** A capable gaming agent needs a world model that improves with the agent,
and an agent that improves with a better world model.

Aligned with CoMAP ([arXiv:2606.02372](https://arxiv.org/abs/2606.02372)), but for
**visual** WM + **VLA** + **sandbox gaming** (not textual LLM agents).

```text
┌─────────────────────────────────────────────┐
│              Sandbox (reality)              │
│         (s,a,r,s') — only ground truth      │
└───────────────┬─────────────┬───────────────┘
                │             │
         update F_φ      update π_θ
                │             │
                v             v
         World Model  ←→   VLA Agent
         imagination      draft→reflect
         + stabilization  + RL
```

## CoMAP → our mapping

| CoMAP | Ours |
|-------|------|
| Textual WM | Action-conditioned visual / latent WM (pretrained_wm) |
| LLM agent | VLA (`π_θ(o,g)→a`) |
| Future-aware reflection | draft → imagine → reliability gate → reflect |
| On-policy self-distillation | student F + EMA teacher + expert mix |
| Env interaction | Sandbox (MineRL) / expert `.pt` proxy |
| — | **Long-horizon stabilization** (phase / re-ground / sparse reality) |

## What is *not* the story anymore

~~Stage1 train WM → Stage2 freeze → train VLA → done~~

That fails because `p_π(s,a) ≠ p_data(s,a)` (exactly CoMAP's critique of fixed WMs).

Stages 1–3 in this package are **warm-up / ablations**, not the final algorithm.
The main algorithm is **`coevolve`**.

## Long-horizon work's place

Hamiltonian / re-grounding / sparse reality become the WM's
**belief stabilization** under a shifting `p_π`:

```
         F_φ
          |
  +-------+-------+
  |               |
dynamics       belief stabilization
(sandbox)      (phase / re-ground / uncertainty)
```

Without this, co-evolution amplifies hallucination instead of skill.

## Run

```bash
cd project_root

# warm-up toy F_φ0
python -m curriculum.run_curriculum pretrain_toy

# CoMAP-style co-evolution (toy π, recommended unit test)
python -m curriculum.run_curriculum coevolve

# ★ Production: base Qwen3-VL base + MineStudio sandbox + WM
export MINESTUDIO_TOKEN=...   # same secret as free_energy MineStudioConfig
python -m curriculum.run_curriculum coevolve_vla

# stub plumbing (no token / no 8B)
python -m curriculum.run_curriculum smoke_coevolve_vla
```

### ARES VLA RL — GRPO (anonymous only; collaborator_b weights read-only)

```bash
cd ares
# hope-test/minestudio_coevolve_wm.hope
# config: .../config_coevolve_wm.py
#   advantage_estimator = "grpo"
#   base: collaborator_b/.../checkpoint-16200
#   token: set MINESTUDIO_TOKEN
#   out:  ares/outputs/openha/grpo_coevolve_wm_openha_32tasks_v2/
```

Sandbox token for the curriculum world-model side: set `MINESTUDIO_TOKEN`.

## Research question (updated)

> How can a world model discover, verify, and consolidate what it knows via
> intervention probes — then later ground a VLA through verified knowledge?

WM↔VLA co-evolution remains the downstream agent loop.

## Directory

```
curriculum/
  auto_research_mvp.py  # ★ Gaming WM Auto Research MVP entry
  research_contract.py  # frozen budget / eval / seeds
  evidence_graph.py     # hypothesis–patch–probe–boundary (+ revoke)
  capability_discovery.py  # auto-propose / verify / compress capabilities
  self_evolve_wm.py     # Discover → Probe → Compress (+ WM update)
  sandbox_experience.py # MineStudio / LLMAPI / stub collectors + interventions
  probes.py             # action / counterfactual / transfer / long-horizon / holdout
  knowledge_memory.py   # compressed verified experience bank
  configs/auto_research_gaming_mvp.yaml
  configs/self_evolve_wm.yaml
  coevolve.py / coevolve_vla.py   # downstream WM↔VLA co-evolution
  configs/coevolve.yaml
  stage1_bundle.py      # warm-up: package frozen pretrained_wm + close-loop
  stage2_vla_rl.py      # ablation: frozen-WM RL
  stage3_wm_adapt.py    # ablation: policy-aware WM only
  stage4_closed_loop.py # ablation: coarse alternating rounds
  frozen_wm.py / policy.py / replay_buffer.py / rewards.py
  vla_bridge.py         # ExternalVLAPolicy for Qwen/ARES
```
