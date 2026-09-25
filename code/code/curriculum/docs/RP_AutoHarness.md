# Auto-Harness

**Question:** given a frozen Qwen-VLA, can an Auto-Harness discover, verify, and improve probe / recover / memory strategies inside a sandbox?

\[
H^*=\arg\max_H J(H;\pi_{\text{VLA}},E)
\]

- **Qwen-VLA:** frozen actor
- **Harness:** the object being optimized
- **LLM:** proposes harness hypotheses
- **Sandbox:** sole ground truth / ACCEPT authority

Orthogonal to `RP_CoEvolve_VLA.md` (trains π) and `RP_DiaWM.md` (trains / diagnoses WM). World model is not layer-1.

---

## 1. Architecture

```text
Observe → Propose → Execute → Evaluate → Improve
```

```text
                 Frozen Qwen-VLA
                       │
                       ▼
                  Auto-Harness
                       │
          ┌────────────┼────────────┐
          │            │            │
        Probe        Verify       Recover
          │            │            │
          └────────────┼────────────┘
                       ▼
                    Sandbox
                       │
             trajectory / reward
                       │
                       ▼
                 Failure Clusters
                       │
                       ▼
                 LLM Proposer
                       │
                  New Harness
                       │
                       ▼
            Multi-seed Evaluation
                       │
                ACCEPT / REVERT
                       │
                       └────→ loop
```

Three roles that must not collapse:

| Role | What |
|------|------|
| **LLM = proposer** | harness diffs only; never `move/attack/jump` |
| **Sandbox = evaluator** | only thing that can ACCEPT |
| **Harness = optimizee** | probe / verify / recover / memory / prompts |

---

## 2. Evaluation protocol (do this first)

Wrong:

\[
J(H)=R(H,\text{seed }101)
\]

Right — same seeds for \(H\) and \(H'\):

\[
\hat J(H)=\frac1K\sum_{i=1}^{K}R(H;s_i)
\qquad
\Delta_i=R(H';s_i)-R(H;s_i)
\]

ACCEPT iff \(\mathrm{mean}_i\Delta_i \ge \varepsilon\). Holdout seeds are disjoint.

**OpenHA caveat:** `eval_seeds` (101/202/…) are paired *episode ids* for harness RNG /
task_id diversity. Minecraft **world seed** comes from `task_config["seed"]`
(often 2025). Overriding world seed breaks `/tp` (black screen). True multi-world
diversity is across tasks, not fake world seeds.

Code: `paired_decision` + `eval_seeds` in `auto_harness_search.py`. Config: `curriculum/configs/auto_harness_minestudio_hf_rl.yaml`.

Editable primitives (beyond `probe.action_pool`): prompts, probe schedule /
`warmup_probes` / `probe_on_stall`, `recover_on_stall`, verify, memory inject,
`runtime.instruction` / `max_steps`. Proposer is steered to multi-axis edits when
seed≈0. LLM API HTTP 429/5xx retries with backoff.

---

## 3. Proposer reliability (do this second)

Truncated Gemini output (`claim` / `reason` then EOF) is **proposer failed**, not **harness failed**.

Contract: `edits` first, `reason` last and short; capture `finish_reason`; retry / repair.

Code: `harness_llm_proposer.py`.

---

## 4. The actual research (only after 2–3)

Not “tune a few thresholds.”

**Can Auto-Harness discover intervention strategies?**

From a small primitive library (risk / future / comparison probes) the loop \(H_0\to H_1\to\cdots\) should invent things like:

- low HP + many enemies → survival probe
- two actions with no progress → replan probe
- memory vs vision conflict → state-verification probe

Later ablations (not now): raw trajectory vs failure clusters; random mutate vs LLM; different probe libraries. World model last, as a data multiplier.

---

## Work order

1. Paired multi-seed \(J(H)\)
2. Proposer JSON / truncation
3. Close `failure → LLM proposal → sandbox → accept/revert`
4. Then clusters / probe types / search methods
5. World model last

One line:

\[
\boxed{\text{Frozen VLA}+\text{Auto-Harness}+\text{Sandbox}}
\qquad
\boxed{\text{LLM proposes, sandbox judges}}
\]

## Run

```bash
cd project_root
python -m curriculum.auto_harness_search --smoke
python -m curriculum.auto_harness_search \
  --config curriculum/configs/auto_harness_minestudio_hf_rl.yaml

# Hard-only Auto (K=1, temp=0, VLM proposer, warm-start prior)
python -u -m curriculum.run_auto_harness_queue \
  --out-root curriculum/outputs/auto_harness_hard_k1_t0 \
  --only-hard-from curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json \
  --wait-bare
```

## Event Auto (HarnessWAM-lite, parallel)

Same protocol as above, but optimize `EventPolicy`
(`event → observe|replan|recover`) with a deterministic compiler.
Does not replace production Auto.

```bash
python -m curriculum.test_auto_harness_event
python -m curriculum.auto_harness_event \
  --config curriculum/configs/auto_harness_event_stub.yaml
python -u -m curriculum.run_auto_harness_event_queue \
  --out-root curriculum/outputs/auto_harness_event_hard_k1_t0 \
  --only-hard-from curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json \
  --wait-bare
```
