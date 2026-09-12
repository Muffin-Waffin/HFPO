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

**Total candidates evaluated:** _(must match total LLM calls across your GA run)_
**Sample size per dataset:** _(same as Exp 1/3 for comparability)_

| Dataset       | Best Random-Search Accuracy | n (candidates evaluated) |
| ------------- | --------------------------: | -----------------------: |
| MedQA         |                    **0.71** |                      100 |
| PubMedQA      |                    **0.78** |                      100 |
| MedMCQA       |                    **0.64** |                  **100** |
| **Aggregate** |                  **0.6812** |                  **100** |

**Notes:**
-

---

## Experiment 3 — Multi-Seed Variance (HFPO)

**Config:** _(generations, population size, held-out set size — fixed across seeds)_

| Seed | MedQA | PubMedQA | MedMCQA | Aggregate | Generation of Best Prompt |
|---|---|---|---|---|---|
| 51 | **68.0%** | **87.0%** | **64.0%** | **73.0%** | 5 |
<!-- | 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| **Mean ± Std/CI** | — | — | — | — | — | -->

**Notes:**

## Per-Generation HFPO Progress (Seed 51)
k
**Config:** Generations, pop=20, elite=2, 100 samples/hospital, Qwen3-8B, temp=0.2, RANDOM_SEED=51

| Generation | Best Fitness | Avg Fitness | Worst Fitness | Time (s) | Best Prompt (truncated) |
|---|---:|---:|---:|---:|---|
| 0 | 0.6867 | 0.6550 | 0.6133 | 3877.2 | Focus on the clinical findings... |
| 1 | 0.7167 | 0.6863 | 0.6400 | 3641.5 | Consider the question from the perspective... |
| 2 | 0.7167 | 0.6783 | 0.5933 | 12073.2 | Consider the question from the perspective... |
| 3 | 0.7267 | 0.6848 | 0.6167 | 1804.9 | Focus on the most critical diagnostic clues... |
| 4 | 0.7267 | 0.6785 | 0.6267 | 2451.9 | Focus on the most critical diagnostic clues... |
| 5 | **0.7300** | 0.6815 | 0.6467 | 1946.6 | **Identify the option that best represents the clinical judgment...** |
| 6 | 0.7300 | 0.6763 | 0.6033 | 1794.7 | Identify the option that best represents the clinical judgment... |
| 7 | 0.7300 | 0.6705 | 0.6000 | 1832.4 | Identify the option that best represents the clinical judgment... |
| 8 | **0.7300** | **0.6952** | 0.6567 | 2275.2 | Identify the option that best represents the clinical judgment... |
<!-- | 8 | 0.7300 | 0.6728 | 0.5767 | 1785.7 | Identify the option that best represents the clinical judgment... |
| 9 | 0.7300 | 0.6752 | 0.6200 | 2924.3 | Identify the option that best represents the clinical judgment... |
| 10 | 0.7300 | 0.6813 | 0.6267 | 5891.3 | Identify the option that best represents the clinical judgment... |
| 11 | 0.7300 | 0.6783 | 0.5833 | 4587.6 | Identify the option that best represents the clinical judgment... |
| 12 | 0.7300 | 0.6837 | 0.6533 | 3157.0 | Identify the option that best represents the clinical judgment... |
| 13 | 0.7300 | 0.6890 | 0.6600 | 1852.8 | Identify the option that best represents the clinical judgment... |
| 14 | 0.7300 | 0.6835 | 0.6400 | 1776.6 | Identify the option that best represents the clinical judgment... |
| 15 | 0.7300 | 0.6843 | 0.6400 | 1778.4 | Identify the option that best represents the clinical judgment... |
| 16 | 0.7300 | 0.6795 | 0.6100 | 1781.4 | Identify the option that best represents the clinical judgment... |
| 17 | 0.7300 | 0.6852 | 0.6367 | 3716.3 | Identify the option that best represents the clinical judgment... |
| 18 | 0.7300 | 0.6943 | 0.6633 | 2329.0 | Identify the option that best represents the clinical judgment... | -->

**Key observations:**
- **Plateau at generation 5** (0-indexed): best fitness reaches 0.7300 and stays flat
- **Mean population fitness** gradually improves from 0.655 → 0.695 despite best staying flat
- **Worst fitness** remains volatile (0.57–0.66), indicating ongoing diversity
- **Total wall-clock**: ~2.2 hours per generation average (varies 30 min – 3.3 hrs)

**Best prompt (found at Gen 5, carried forward as elite):**
> `Identify the option that best represents the clinical judgment of an experienced practitioner, emphasizing evidence-based decision-making and the avoidance of misleading or irrelevant choices.`

**Per-hospital breakdown at Generations (final):**
| Hospital | Score |
|---|---|
| MedQA | 68.0% |
| PubMedQA | 87.0% |
| MedMCQA | 64.0% |
| **Aggregate** | **73.0%** 

---

##  Centralized vs. Federated Evaluation Ablation

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

## Multi-Model Comparison

| Model | Zero-shot (Expert) | Zero-shot (CoT) | Best Evolved (HFPO) | Absolute Gain (vs CoT) | Generation of Best Prompt |
|---|---|---|---|---|---|
| Qwen3-8B | — | — | — | — | — |
| PHI-4-14b | — | — | — | — | — |
| GPT-OSS-20b | — | — | — | — | — |

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

## Experiment 12 — Per-Generation HFPO Progress (Seed 51)

**Config:** 20 generations, pop=20, elite=2, 100 samples/hospital, Qwen3-8B, temp=0.2, RANDOM_SEED=51

| Generation | Best Fitness | Avg Fitness | Worst Fitness | Time (s) | Best Prompt (truncated) |
|---|---:|---:|---:|---:|---|
| 0 | 0.6867 | 0.6550 | 0.6133 | 3877.2 | Focus on the clinical findings... |
| 1 | 0.7167 | 0.6863 | 0.6400 | 3641.5 | Consider the question from the perspective... |
| 2 | 0.7167 | 0.6783 | 0.5933 | 12073.2 | Consider the question from the perspective... |
| 3 | 0.7267 | 0.6848 | 0.6167 | 1804.9 | Focus on the most critical diagnostic clues... |
| 4 | 0.7267 | 0.6785 | 0.6267 | 2451.9 | Focus on the most critical diagnostic clues... |
| 5 | **0.7300** | 0.6815 | 0.6467 | 1946.6 | **Identify the option that best represents the clinical judgment...** |
| 6 | 0.7300 | 0.6763 | 0.6033 | 1794.7 | Identify the option that best represents the clinical judgment... |
| 7 | 0.7300 | 0.6705 | 0.6000 | 1832.4 | Identify the option that best represents the clinical judgment... |
| 8 | 0.7300 | 0.6728 | 0.5767 | 1785.7 | Identify the option that best represents the clinical judgment... |
| 9 | 0.7300 | 0.6752 | 0.6200 | 2924.3 | Identify the option that best represents the clinical judgment... |
| 10 | 0.7300 | 0.6813 | 0.6267 | 5891.3 | Identify the option that best represents the clinical judgment... |
| 11 | 0.7300 | 0.6783 | 0.5833 | 4587.6 | Identify the option that best represents the clinical judgment... |
| 12 | 0.7300 | 0.6837 | 0.6533 | 3157.0 | Identify the option that best represents the clinical judgment... |
| 13 | 0.7300 | 0.6890 | 0.6600 | 1852.8 | Identify the option that best represents the clinical judgment... |
| 14 | 0.7300 | 0.6835 | 0.6400 | 1776.6 | Identify the option that best represents the clinical judgment... |
| 15 | 0.7300 | 0.6843 | 0.6400 | 1778.4 | Identify the option that best represents the clinical judgment... |
| 16 | 0.7300 | 0.6795 | 0.6100 | 1781.4 | Identify the option that best represents the clinical judgment... |
| 17 | 0.7300 | 0.6852 | 0.6367 | 3716.3 | Identify the option that best represents the clinical judgment... |
| 18 | 0.7300 | 0.6943 | 0.6633 | 2329.0 | Identify the option that best represents the clinical judgment... |
| 19 | 0.7300 | 0.6952 | 0.6567 | 2275.2 | Identify the option that best represents the clinical judgment... |

**Key observations:**
- **Plateau at generation 5** (0-indexed): best fitness reaches 0.7300 and stays flat through gen 19
- **Mean population fitness** gradually improves from 0.655 → 0.695 despite best staying flat
- **Worst fitness** remains volatile (0.57–0.66), indicating ongoing diversity
- **Total wall-clock**: ~2.2 hours per generation average (varies 30 min – 3.3 hrs)

**Best prompt (found at Gen 5, carried forward as elite):**
> `Identify the option that best represents the clinical judgment of an experienced practitioner, emphasizing evidence-based decision-making and the avoidance of misleading or irrelevant choices.`

**Per-hospital breakdown at Gen 19 (final):**
| Hospital | Score |
|---|---|
| MedQA | 68.0% |
| PubMedQA | 87.0% |
| MedMCQA | 64.0% |
| **Aggregate** | **73.0%** |

---

## Master Summary Table (for paper Table 1 — Qwen3-8B)

| Method | MedQA | PubMedQA | MedMCQA | Avg (± CI) |
|---|---|---|---|---|
| Zero-shot (Expert) | — | — | — | — |
| Zero-shot (CoT) | — | — | — | — |
| Random search (compute-matched) | — | — | — | — |
| Vanilla EvoPrompt (centralized) | — | — | — | — |
| HFPO — federated, secure agg | — | — | — | — |
<!-- | HFPO — centralized eval | — | — | — | — | -->
<!-- | HFPO — federated, no privacy | — | — | — | — | -->
<!-- | HFPO — federated, DP (ε = ___) | — | — | — | — | -->

---

*Last updated: 2026-07-13 (added Exp 12 per-generation HFPO progress from completed 20-gen run)*
