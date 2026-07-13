# HFPO Experiment Results Tracker

Fill in values as experiments complete. Leave blank cells as `—` until data is available.
Report accuracy as percentages with 1 decimal place. Where noted, include ± 95% CI or std dev.

---

## Experiment 1 — Zero-shot Baseline (Qwen3-8B)

**Prompts tested:** `expert` (naive floor), `cot` (chain-of-thought floor)
**Sample size per dataset:** _(fill in — target 100+ per dataset, held-out/full split)_
**Aggregation method used:** _(macro-avg / micro-avg — must match Exp 3/HFPO fitness method)_

| Dataset | Expert Accuracy | CoT Accuracy | n (samples) |
|---|---|---|---|
| MedQA | 60.00% | 67.33% | 300 |
| PubMedQA | 61.33% | 69.67% | 300 |
| MedMCQA | 64.00% | 68.33% | 300 |
|  | **61.78%** | **68.44%** | 900 total |

**Notes / anomalies:**
- CoT beats expert on all 3 datasets consistently (+4.3 to +8.3 pts) — resolves earlier n=10/n=20 smoke-test noise where MedQA briefly showed CoT underperforming.
- CoT parser confirmed 100% answer_anchor match rate at n=20 smoke test after MAX_NEW_TOKENS_COT raised to 1536 (no fallback/truncation contamination). Not yet re-confirmed at n=300 — worth a quick check of cot_debug_log.jsonl fallback rate before treating this as fully final.
- Expert-prompt MedQA score (60.0%) is notably below published Qwen3-8B zero-shot MedQA (~74.2%, MedPRMBench). Worth a one-line caveat in the paper, or a quick check whether expert's persona framing underperforms a truly bare-minimum prompt.

---

## Experiment 2 — Random Search Baseline (compute-matched)

**Total candidates evaluated:** _(must match total LLM calls across your 20-gen GA run)_
**Sample size per dataset:** _(same as Exp 1/3 for comparability)_

| Dataset | Best Random-Search Accuracy | n (candidates evaluated) |
|---|---|---|
| MedQA | — | — |
| PubMedQA | — | — |
| MedMCQA | — | — |
| **Aggregate** | — | — |

**Notes:**
-

---

## Experiment 3 — Multi-Seed Variance (HFPO, post-bugfix)

**Config:** _(generations, population size, held-out set size — fixed across seeds)_

| Seed | MedQA | PubMedQA | MedMCQA | Aggregate | Generation of Best Prompt |
|---|---|---|---|---|---|
| 1 (existing run) | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| **Mean ± Std/CI** | — | — | — | — | — |

**Notes:**
-

---

## Experiment 4 — Centralized vs. Federated Evaluation Ablation

| Condition | MedQA | PubMedQA | MedMCQA | Aggregate | Notes |
|---|---|---|---|---|---|
| Centralized (pooled) eval | — | — | — | — | |
| Federated (per-client) eval | — | — | — | — | |
| **Δ (Federated − Centralized)** | — | — | — | — | |

---

## Experiment 5 — Privacy-Budget Accuracy Tradeoff

| Privacy Setting | MedQA | PubMedQA | MedMCQA | Aggregate |
|---|---|---|---|---|
| No privacy | — | — | — | — |
| DP (ε = ___) | — | — | — | — |
| DP (ε = ___) | — | — | — | — |
| DP (ε = ___) | — | — | — | — |
| Secure aggregation | — | — | — | — |

*(Plot this as a line/curve: aggregate accuracy vs. privacy setting, for the paper figure.)*

---

## Experiment 6 — Vanilla EvoPrompt Reimplementation (centralized, no federation)

| Dataset | Vanilla EvoPrompt Accuracy | HFPO (centralized, Exp 4 row) | Δ |
|---|---|---|---|
| MedQA | — | — | — |
| PubMedQA | — | — | — |
| MedMCQA | — | — | — |
| **Aggregate** | — | — | — |

---

## Experiment 7 — Crossover-Bugfix Ablation

| Config | MedQA | PubMedQA | MedMCQA | Aggregate | Generation Plateau Observed |
|---|---|---|---|---|---|
| Pre-fix | — | — | — | — | — |
| Post-fix | — | — | — | — | — |

---

## Experiment 8 — Cross-Dataset Generalization

Best-evolved prompt from primary run, evaluated (inference-only) on datasets it wasn't evolved on.

| Prompt Evolved On | Evaluated On | Accuracy | Native (in-domain) Accuracy | Δ (Transfer Gap) |
|---|---|---|---|---|
| MedQA | PubMedQA | — | — | — |
| MedQA | MedMCQA | — | — | — |
| (repeat if evolving separately per dataset) | | | | |

---

## Experiment 9 — Multi-Model Comparison

| Model | Zero-shot (Expert) | Zero-shot (CoT) | Best Evolved (HFPO) | Absolute Gain (vs CoT) | Generation of Best Prompt |
|---|---|---|---|---|---|
| Qwen3-8B | — | — | — | — | — |
| Meditron3-8B | — | — | — | — | — |
| Llama-3.1-8B | — | — | — | — | — |

*(Figure: best fitness vs. generation, one curve per model.)*
*(Optional Figure: population diversity vs. generation, one curve per model — requires diversity metric logging.)*

---

## Experiment 10 — Client-Count Scaling (Tier 3)

| # Clients | MedQA | PubMedQA | MedMCQA | Aggregate |
|---|---|---|---|---|
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| 5 | — | — | — | — |
| ... | — | — | — | — |

---

## Experiment 11 — Diversity-Collapse Intervention (Tier 3)

| Config | Generation Plateau Observed | Final Aggregate Accuracy | Diversity Metric @ Gen 8 |
|---|---|---|---|
| Baseline (no intervention) | — | — | — |
| With intervention (___) | — | — | — |

---

## Master Summary Table (for paper Table 1 — Qwen3-8B)

| Method | MedQA | PubMedQA | MedMCQA | Avg (± CI) |
|---|---|---|---|---|
| Zero-shot (Expert) | — | — | — | — |
| Zero-shot (CoT) | — | — | — | — |
| Random search (compute-matched) | — | — | — | — |
| Vanilla EvoPrompt (centralized) | — | — | — | — |
| HFPO — centralized eval | — | — | — | — |
| HFPO — federated, no privacy | — | — | — | — |
| HFPO — federated, DP (ε = ___) | — | — | — | — |
| HFPO — federated, secure agg | — | — | — | — |

---

*Last updated: — (update this line each time a table is filled in)*
