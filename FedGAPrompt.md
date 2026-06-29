# FedGAPrompt v3 — Implementation Plan (Revised)
### Federated LLM-Guided Evolutionary Prompt Optimization (HFPO)

---

## Changelog from v1

| # | Change | Reason |
|---|---|---|
| 1 | Tournament selection replaces top-k cutoff | Preserves diversity; prevents best prompt dominating every generation |
| 2 | Weaker prompts are parent candidates too | Tournament sampling, not ranking cutoff — all 20 prompts compete |
| 3 | Prompt Generator receives richer context | Smarter variation; better-reasoned children |
| 4 | Batch generation (6 children per LLM call) | Cheaper and faster than 12 serial calls |
| 5 | Deployment prompt = highest evaluated prompt | Unevaluated synthesis is indefensible to reviewers |
| 6 | Full lineage tree, not just parent_ids | Enables evolutionary tree figure for paper |
| 7 | Evaluation cache | Skip re-evaluation of duplicate prompt texts |
| 8 | Ablation experiment added | Justifies the GA; answers "why not just call the LLM once?" |
| 9 | Renamed LLMVariationOperator → PromptGenerator | Architecturally accurate; easier to defend in paper |

---

## 1. Architectural Audit — What Exists vs. What Changes

### 1.1 Files/Modules That Remain Unchanged

| Component | Reason |
|---|---|
| `model_loader.py` | Infrastructure; not touched by redesign |
| `dataset_loader.py` | Hospitals still load local data the same way |
| `evaluation_pipeline.py` | Core scoring logic is fully reused |
| `prompt_scorer.py` | Operates on prompt + dataset; unchanged |
| `seed_prompts.py` | 20 seed prompts are still the starting point |

---

### 1.2 Files That Must Be Modified

| Component | What Changes |
|---|---|
| `prompt_candidate.py` | Scalar fitness → FitnessVector; add lineage fields |
| `population_manager.py` | Operate on FitnessVectors; integrate evaluation cache |
| `genetic_algorithm.py` | Remove top-k cutoff; implement tournament selection |
| `federated_client.py` | Strip all evolution logic; pure evaluator only |
| `federated_server.py` | Own the full evolutionary loop |
| `run_experiment.py` | Rewire main loop; add ablation experiment mode |
| `config.py` | Add new configuration keys |

---

### 1.3 Classes/Modules to Remove

| Component | Why |
|---|---|
| Handcrafted mutation operators | Replaced by PromptGenerator |
| String-based crossover | Replaced by semantic batch generation |
| All `evolve()` / `mutate()` methods in Hospital | Hospitals are evaluators only |
| `LocalFitness: float` scalar | Superseded by FitnessVector |

---

### 1.4 New Classes/Modules to Create

| Component | Responsibility |
|---|---|
| `FitnessVector` | Per-hospital scores + derived metrics (average, min, variance) |
| `LineageTracker` | Records parent → child chains; builds full ancestry tree |
| `PromptGenerator` | Wraps reasoning LLM; batch mutation + batch crossover with rich context |
| `TournamentSelector` | GA tournament selection; replaces top-k cutoff |
| `ServerEvolutionEngine` | Owns selection, generation, assembly, orchestration |
| `EvaluationCache` | Hash prompt text → FitnessVector; skip duplicate evaluations |
| `GenerationLogger` | Snapshot serialization per generation |
| `OutputManager` | Derives the three final outputs |
| `AblationRunner` | Runs LLM-only baseline for comparison |

---

## 2. Redesigned Data Structures

### 2.1 FitnessVector

```
FitnessVector
├── scores: dict[str, float]       # {"hospital_A": 0.81, "hospital_B": 0.87, "hospital_C": 0.79}
├── average() → float
├── minimum() → float
├── variance() → float
├── worst_case() → float           # Alias for minimum(); used in robustness ranking
└── best_hospital() → str          # Hospital ID with highest score
    worst_hospital() → str         # Hospital ID with lowest score
```

The vector is **never collapsed** during the evolutionary process. Derived scalars are computed only when a ranking signal is needed.

---

### 2.2 PromptCandidate (Redesigned)

```
PromptCandidate
├── id: str                          # Unique ID (uuid4)
├── text: str                        # The prompt string
├── generation: int                  # Which generation produced this prompt
├── parent_ids: list[str]            # Direct parents (1 for mutation, 2 for crossover)
├── ancestry_ids: list[str]          # Full lineage back to seed (for evolutionary tree)
├── origin: str                      # "seed" | "survivor" | "mutation" | "crossover"
├── fitness: FitnessVector | None    # Populated after evaluation; None before
└── metadata: dict                   # Arbitrary logging (e.g. tournament rank, LLM call ID)
```

`ancestry_ids` is computed by `LineageTracker` at creation time — it merges and deduplicates the ancestry chains of all parent prompts.

---

### 2.3 GenerationSnapshot

```
GenerationSnapshot
├── generation_id: int
├── population: list[PromptCandidate]   # Always exactly 20
├── tournament_results: list[dict]      # Which prompts competed in each tournament round
├── selected_parents: list[str]         # IDs of prompts selected as parents this generation
└── timestamp: str
```

---

### 2.4 EvaluationCache

```
EvaluationCache
├── _store: dict[str, FitnessVector]   # key = SHA256(prompt.text)
├── get(prompt_text) → FitnessVector | None
├── set(prompt_text, vector)
└── size() → int
```

Before evaluating any prompt, the server checks the cache. If the exact text has been evaluated before, the stored vector is attached and evaluation is skipped for that prompt across all hospitals.

---

## 3. Communication Flow — Server ↔ Hospitals

### 3.1 Broadcast (Server → Hospitals)

```json
{
  "generation": 3,
  "prompts": [
    {"id": "abc123", "text": "Answer the following medical question..."},
    ...
  ]
}
```

Hospitals receive only `id` and `text`. No fitness vectors are shared between hospitals.

---

### 3.2 Evaluation Response (Hospital → Server)

```json
{
  "hospital_id": "hospital_A",
  "generation": 3,
  "results": [
    {"prompt_id": "abc123", "score": 0.81},
    ...
  ]
}
```

Hospitals return scores only. The server owns all aggregation, selection, and generation logic.

---

### 3.3 Cache-Filtered Broadcast

Before broadcasting, the server checks `EvaluationCache` for each prompt:

- **Cache hit:** attach stored `FitnessVector` immediately; exclude from broadcast
- **Cache miss:** include in broadcast; evaluate normally

After receiving results, the server stores new vectors in the cache.

---

## 4. Tournament Selection — Replacing Top-K Cutoff

Tournament selection preserves diversity by giving every prompt a chance to become a parent, rather than hard-cutting the bottom 12.

### 4.1 Algorithm

```
To select one parent:
  1. Sample k prompts at random from the population (tournament_size, default k=4)
  2. The prompt with the highest fitness among those k is selected as parent
  3. Return the winner

Repeat until enough parents are selected for all 12 children.
(With replacement — the same prompt can be selected as parent multiple times)
```

### 4.2 Why This Matters

With top-8 cutoff:
- If Generation 3's best prompt scores 0.91 average, it will dominate every parent slot
- Within 3–4 generations, the population converges to variations of one prompt
- Diversity collapses

With tournament selection:
- A prompt scoring 0.72 can still win its tournament round by chance
- Weaker prompts that excel on one hospital axis can still contribute lineage
- Diversity is maintained across generations

### 4.3 TournamentSelector Interface

```
TournamentSelector
├── tournament_size: int = 4
├── select_parent(population) → PromptCandidate
└── select_parents(population, n_parents) → list[PromptCandidate]
```

`select_parents(population, n_parents=24)` — over-selects slightly so crossover pairings have enough variety.

---

## 5. Prompt Generator — Redesigned with Rich Context

The component formerly called `LLMVariationOperator` is renamed `PromptGenerator`.

**Conceptual framing:**
- The GA selects which prompts deserve to survive and become parents
- The Prompt Generator creates new candidate prompts using those parents as inspiration
- These are distinct responsibilities; the naming should reflect that

---

### 5.1 Context Payload Sent to the Reasoning LLM

Every call to the Prompt Generator includes:

```
{
  "task_description":      "Medical question answering on clinical datasets",
  "current_generation":    5,
  "current_best_prompt":   "...",
  "current_best_average":  0.89,
  "parent_prompt":         "...",
  "parent_fitness": {
      "hospital_A": 0.81,
      "hospital_B": 0.87,
      "hospital_C": 0.79,
      "average":    0.82,
      "best_hospital":  "hospital_B",
      "worst_hospital": "hospital_C"
  }
}
```

For crossover, `parent_prompt_B` and its fitness vector are added.

---

### 5.2 Mutation Prompt Template

```
You are optimizing prompts for a medical question-answering task.
Task: {task_description}
This is generation {current_generation} of an evolutionary optimization process.

Current best prompt in the population (average score: {current_best_average:.0%}):
"{current_best_prompt}"

The following parent prompt achieved these scores across three hospital datasets:
  Hospital A: {score_A:.0%}
  Hospital B: {score_B:.0%}
  Hospital C: {score_C:.0%}
  Average: {average:.0%}
  Strongest hospital: {best_hospital}
  Weakest hospital:   {worst_hospital}

Parent prompt:
"{parent_text}"

Generate {n} new candidate prompts that:
1. Address the weakness on {worst_hospital}
2. Maintain the strengths that produced the {best_hospital} score
3. Are semantically distinct from the parent — not paraphrases

Return a JSON array of {n} prompt strings. No explanation. No keys. Only the array.
```

---

### 5.3 Crossover Prompt Template

```
You are optimizing prompts for a medical question-answering task.
Task: {task_description}
This is generation {current_generation} of an evolutionary optimization process.

Current best prompt in the population (average score: {current_best_average:.0%}):
"{current_best_prompt}"

Parent A (scores — A: {a_A:.0%}, B: {a_B:.0%}, C: {a_C:.0%}, avg: {a_avg:.0%}):
"{text_A}"

Parent B (scores — A: {b_A:.0%}, B: {b_B:.0%}, C: {b_C:.0%}, avg: {b_avg:.0%}):
"{text_B}"

Generate {n} new candidate prompts that combine the strongest qualities of both parents.
The results should outperform both parents across all three hospital datasets.

Return a JSON array of {n} prompt strings. No explanation. No keys. Only the array.
```

---

### 5.4 Batch Generation Strategy

Instead of 12 serial LLM calls (one child each), use 2 batch calls:

```
Batch 1: Select 2 parents via tournament → call LLM → generate 6 children (crossover)
Batch 2: Select 2 parents via tournament → call LLM → generate 6 children (mutation or crossover)

Total: 2 LLM calls → 12 children
```

The LLM returns a JSON array of 6 prompt strings per call. The server parses and wraps each into a `PromptCandidate`.

This is 6× cheaper and removes the latency of 10 extra round-trips.

---

### 5.5 PromptGenerator Interface

```
PromptGenerator
├── model: ReasoningLLM
├── task_description: str
│
├── generate_mutations(parent, n, context) → list[PromptCandidate]
├── generate_crossovers(parent_a, parent_b, n, context) → list[PromptCandidate]
├── batch_generate(parents, n_children, context) → list[PromptCandidate]
│       Schedules batch calls; handles JSON parsing; validates output count
│
└── _call_llm(prompt_text: str) → list[str]
        Returns list of n prompt strings; retries on malformed JSON
```

---

## 6. Server Evolution Engine — Full Design

```
ServerEvolutionEngine
├── tournament_selector: TournamentSelector
├── prompt_generator: PromptGenerator
├── evaluation_cache: EvaluationCache
├── lineage_tracker: LineageTracker
│
├── broadcast(population) → None
├── collect_results() → list[HospitalResponse]
├── aggregate(population, results) → list[PromptCandidate]
│       Attaches FitnessVectors; checks cache first
│
├── select_parents(population, n) → list[PromptCandidate]
│       Delegates to TournamentSelector
│
├── generate_children(parents, n_children) → list[PromptCandidate]
│       Delegates to PromptGenerator.batch_generate
│
├── build_next_generation(current_population, children) → list[PromptCandidate]
│       Carries survivors forward + adds children
│       NOTE: "survivors" here means prompts not replaced, not a hard top-k cutoff
│       Configurable: can use elitism (top N always survive) + tournament for the rest
│
└── run_generation(population, generation_id) → GenerationSnapshot
        Full orchestration of one complete generation
```

---

## 7. Lineage Tracker

```
LineageTracker
├── _tree: dict[str, list[str]]          # prompt_id → list[parent_ids]
│
├── register(prompt: PromptCandidate)
│       Stores id → parent_ids mapping; computes ancestry_ids from parents' records
│
├── get_ancestry(prompt_id) → list[str]
│       Returns full chain: [grandparent, parent, self] back to seed
│
├── get_depth(prompt_id) → int
│       Generation depth from seed
│
└── export_tree() → dict
        Serializable structure for paper figures (nodes + edges)
```

The evolutionary tree figure for the paper is produced directly from `export_tree()`.

---

## 8. Hospital (Client) — Redesigned Interface

```
HospitalClient
├── hospital_id: str
├── local_dataset: Dataset               # Loaded once; never shared
│
└── evaluate_population(prompts: list[dict]) → list[dict]
        Input:  [{"id": str, "text": str}, ...]
        Output: [{"prompt_id": str, "score": float}, ...]
        Uses:   existing evaluation_pipeline + prompt_scorer unchanged
```

All evolution methods removed. The hospital is a pure evaluation oracle.

---

## 9. Output Manager — Three Final Outputs (Revised)

### 9.1 Local Prompt (Per Hospital)

For each hospital, scan all `GenerationSnapshot` records. Find the `PromptCandidate` whose `FitnessVector.scores[hospital_id]` is highest across all generations.

```
OutputManager.compute_local_best(history) → dict[str, PromptCandidate]
  Returns: {"hospital_A": best_for_A, "hospital_B": best_for_B, "hospital_C": best_for_C}
```

---

### 9.2 Federated Prompt

The prompt with the highest **average** `FitnessVector` across all generations and all hospitals.

```
OutputManager.compute_federated_best(history) → PromptCandidate
```

---

### 9.3 Deployment Prompt *(revised)*

**The deployment prompt is the federated best prompt.**

It is not a newly generated LLM synthesis. An unevaluated synthesis cannot be defended as the deployment output — reviewers will immediately question its validity since it was never tested.

```
OutputManager.compute_deployment_prompt(history) → PromptCandidate
  Returns: same result as compute_federated_best(history)
  Reason:  highest-evaluated prompt across the entire experiment
```

**Optional extension (if synthesis is desired for analysis):**
If an LLM synthesis is generated for qualitative comparison, it must be subsequently evaluated through one additional federated round before it can be reported as a deployment candidate. This is noted as future work.

---

## 10. Ablation Experiment — LLM Only vs. GA + LLM

This is the most important experiment in the paper. Without it, reviewers will ask: *"Why do you need the GA at all?"*

### 10.1 Baseline: LLM-Only Condition

```
AblationRunner.run_llm_only(seed_prompts, n_eval_rounds)

1. Start with same 20 seed prompts
2. Call PromptGenerator once (no selection, no tournament, no fitness-informed context)
3. Generate 20 new prompts
4. Evaluate all 20 across all hospitals
5. Report best result
```

No iterative refinement. No fitness feedback. Just one round of generation + evaluation.

### 10.2 GA + LLM Condition

Normal FedGAPrompt run: 10 generations, tournament selection, fitness-informed generation.

### 10.3 Metrics to Compare

| Metric | LLM Only | GA + LLM |
|---|---|---|
| Best average fitness | — | — |
| Best per-hospital fitness | — | — |
| Variance across hospitals | — | — |
| Robustness (min fitness) | — | — |

### 10.4 `AblationRunner` Interface

```
AblationRunner
├── run_llm_only(seed_prompts, hospitals) → AblationResult
├── run_ga_llm(seed_prompts, hospitals, generations) → AblationResult
└── compare(result_a, result_b) → ComparisonReport
```

---

## 11. Lifecycle of a Prompt Through One Generation

```
[GENERATION START]
│
├─ Population: 20 PromptCandidates (all fitness vectors populated from previous generation)
│
│  ── CACHE CHECK ─────────────────────────────────────────────────────────
│
├─ For each prompt: check EvaluationCache
│     Cache hit  → attach stored FitnessVector; exclude from broadcast
│     Cache miss → include in broadcast queue
│
│  ── BROADCAST ──────────────────────────────────────────────────────────
│
├─ Server sends uncached prompt texts to Hospital A, B, C
│
│  ── EVALUATION ─────────────────────────────────────────────────────────
│
├─ Hospital A evaluates → returns scores
├─ Hospital B evaluates → returns scores
├─ Hospital C evaluates → returns scores
│
│  ── AGGREGATION ────────────────────────────────────────────────────────
│
├─ Server assembles FitnessVector per prompt
│     e.g. Prompt X → {A: 0.81, B: 0.87, C: 0.79, avg: 0.82}
├─ New vectors stored in EvaluationCache
│
│  ── TOURNAMENT SELECTION ───────────────────────────────────────────────
│
├─ TournamentSelector samples from all 20 prompts
│     Each tournament: sample 4 prompts → best wins → selected as parent
│     Repeat until enough parents selected for 12 children
│     (Replacement allowed — strong prompts can be selected multiple times)
│
│  ── BATCH GENERATION ───────────────────────────────────────────────────
│
├─ PromptGenerator.batch_generate(parents, n_children=12, context)
│     Context includes: task description, generation number,
│                       current best prompt + score, each parent's fitness vector
│
│  ┌─ Batch Call 1 (LLM → 6 children)
│  │   Input:  parent_A + parent_B + rich context
│  │   Output: ["new prompt 1", ..., "new prompt 6"]
│  │
│  └─ Batch Call 2 (LLM → 6 children)
│      Input:  parent_C + parent_D + rich context
│      Output: ["new prompt 7", ..., "new prompt 12"]
│
├─ Each child string wrapped into PromptCandidate
│     origin = "mutation" or "crossover"
│     parent_ids = [parent_A.id, parent_B.id]
│     ancestry_ids = LineageTracker.get_ancestry(parent_A) ∪ get_ancestry(parent_B)
│     fitness = None (populated next generation)
│
│  ── ASSEMBLY ───────────────────────────────────────────────────────────
│
├─ Next generation: carry top-N survivors (elitism) + 12 new children = 20 prompts
│   NOTE: Elitism here means the highest-fitness prompts from current generation
│         always survive — they are not re-evaluated, just carried forward
│
│  ── LOGGING ────────────────────────────────────────────────────────────
│
└─ GenerationLogger saves snapshot → [NEXT GENERATION START]
```

---

## 12. Implementation Roadmap — Eight Milestones

Each milestone is independently testable before proceeding.

---

### Milestone 1 — Data Structure Redesign
**Scope:** Replace scalar fitness with FitnessVector; update PromptCandidate with lineage fields.

**Tasks:**
1. Create `fitness_vector.py` — `FitnessVector` dataclass with `scores` dict; implement `average()`, `minimum()`, `variance()`, `worst_case()`, `best_hospital()`, `worst_hospital()`.
2. Update `PromptCandidate` — remove scalar fitness; add `fitness: FitnessVector | None`, `origin`, `parent_ids`, `ancestry_ids`, `generation`.
3. Create `lineage_tracker.py` — implement `register()`, `get_ancestry()`, `get_depth()`, `export_tree()`.
4. Create `evaluation_cache.py` — implement SHA256-keyed cache with `get()`, `set()`, `size()`.

**Test:** Create 20 `PromptCandidate` objects with mock `FitnessVector` data. Verify sorting by average, minimum, variance. Register them in `LineageTracker` with mock parent chains; verify `get_ancestry()` returns correct depth. Verify `EvaluationCache` returns hits correctly. No model inference needed.

---

### Milestone 2 — Hospital Refactor
**Scope:** Strip all evolution logic from `HospitalClient`; make it a pure evaluator.

**Tasks:**
1. Delete `evolve()`, `mutate()`, `crossover()`, `select()` methods from `HospitalClient`.
2. Implement `evaluate_population(prompts: list[dict]) → list[dict]` using existing evaluation pipeline and prompt scorer.
3. Confirm hospitals never modify prompt text.
4. Confirm `hospital_id` is present in every response payload.

**Test:** Pass 20 hardcoded prompts to one `HospitalClient`. Confirm exactly 20 scores returned. Confirm no prompt text is modified. Confirm `hospital_id` is in every result. Uses real model + dataset loading.

---

### Milestone 3 — Server Aggregation with Cache
**Scope:** Server collects results, assembles FitnessVectors, checks/updates EvaluationCache.

**Tasks:**
1. Implement `aggregate_results(population, raw_results) → list[PromptCandidate]`.
2. Before broadcasting: check cache for each prompt; exclude cache-hits from broadcast.
3. After receiving results: assemble FitnessVectors; store new entries in cache.
4. Add validation: every uncached prompt must have a result from every hospital.

**Test:** Feed mock hospital responses into `aggregate_results`. Confirm every `PromptCandidate.fitness` is a valid `FitnessVector`. Re-run with the same prompts; confirm cache hits are returned immediately with no hospital calls. Confirm mismatched IDs raise clear errors.

---

### Milestone 4 — Tournament Selection
**Scope:** Implement `TournamentSelector`; wire into `ServerEvolutionEngine`.

**Tasks:**
1. Create `tournament_selector.py` — implement `select_parent(population)` (samples `tournament_size` prompts; returns winner by `fitness.average()`).
2. Implement `select_parents(population, n_parents)` — calls `select_parent` n times with replacement.
3. Add `tournament_size` to `config.py` (default 4).
4. Add unit test for diversity: confirm that across 1000 tournament runs on a 20-prompt population, prompts ranking outside the top 5 are selected at least occasionally.

**Test:** Mock population of 20 prompts with varied fitness vectors. Confirm `select_parents(population, 24)` always returns exactly 24. Confirm diversity: the lowest-fitness prompt is selected at least once in 1000 trials. No LLM needed.

---

### Milestone 5 — Prompt Generator (Batch)
**Scope:** Implement `PromptGenerator` with rich context and batch generation.

**Tasks:**
1. Create `prompt_generator.py`.
2. Implement `_call_llm(prompt_text) → list[str]` — returns parsed JSON array of strings; retries on malformed JSON (max 2 retries).
3. Implement `generate_mutations(parent, n, context)` using mutation template from Section 5.2.
4. Implement `generate_crossovers(parent_a, parent_b, n, context)` using crossover template from Section 5.3.
5. Implement `batch_generate(parents, n_children=12, context)` — schedules 2 LLM calls of 6 children each; wraps results into `PromptCandidate` objects with correct `origin`, `parent_ids`, `ancestry_ids`.
6. Add deduplication: if a generated text matches any existing prompt in the cache, retry that slot once.

**Test:** Run `batch_generate` on 4 mock survivors. Confirm exactly 12 children returned. Confirm all have non-empty text, valid `parent_ids`, correct `origin`, non-None `ancestry_ids`. Inspect LLM outputs qualitatively for semantic diversity.

---

### Milestone 6 — Full Single-Generation Loop
**Scope:** Wire Milestones 1–5 into one complete generation cycle.

**Tasks:**
1. Implement `ServerEvolutionEngine.run_generation(population, generation_id) → GenerationSnapshot`.
2. Sequence: cache check → broadcast → hospital evaluation → aggregate → tournament selection → batch generation → assembly → log.
3. Implement `GenerationLogger.save(snapshot)` — writes to JSON.
4. Implement `build_next_generation(current_pop, children)` — elitism for top prompts + 12 new children = 20 total.

**Test:** Run exactly 1 generation end-to-end with 20 seed prompts and 3 real hospitals. Confirm output population = exactly 20. Confirm `GenerationSnapshot` is written to disk. Confirm `EvaluationCache` now contains 20 entries. Inspect fitness vectors.

---

### Milestone 7 — Multi-Generation Federated Loop
**Scope:** Run N generations; accumulate `GenerationSnapshot` history.

**Tasks:**
1. Implement main training loop in `run_experiment.py`.
2. Initialize seed population (generation 0, no fitness yet).
3. For each generation: run `run_generation()`, append snapshot to history.
4. Console logging: print best average fitness per generation.
5. Checkpoint saving: persist current population every N generations.
6. Stop condition: `max_generations` (configurable); optional: early stop if best fitness doesn't improve for K consecutive generations.

**Test:** Run 3 full generations. Confirm fitness improves or is stable. Confirm `LineageTracker.export_tree()` produces a valid tree structure with correct depth. Confirm all snapshots logged. Confirm cache grows across generations and reduces evaluation calls.

---

### Milestone 8 — Outputs + Ablation Experiment
**Scope:** Derive the three final outputs; implement and run the ablation.

**Tasks:**
1. Implement `OutputManager.compute_local_best(history)` — highest per-hospital score across all generations.
2. Implement `OutputManager.compute_federated_best(history)` — highest average across all generations.
3. Implement `OutputManager.compute_deployment_prompt(history)` — returns `compute_federated_best(history)` (highest evaluated prompt; not LLM synthesis).
4. Implement `AblationRunner.run_llm_only(seed_prompts, hospitals)` — one-shot generation, no selection, no fitness context.
5. Implement `AblationRunner.compare(result_a, result_b) → ComparisonReport`.
6. Save all outputs to structured JSON + human-readable summary.

**Test:** Run on complete history from Milestone 7. Confirm three distinct outputs produced. Confirm local bests are different per hospital (expected under non-IID data). Run `AblationRunner`; confirm LLM-only and GA+LLM produce different results. Confirm comparison report is interpretable.

---

## 13. Configuration Additions

```python
# Evolution
POPULATION_SIZE         = 20
MAX_GENERATIONS         = 10
EARLY_STOP_PATIENCE     = 3        # Stop if no improvement for 3 generations
RANKING_STRATEGY        = "average"  # "average" | "minimum" | "weighted"

# Tournament Selection
TOURNAMENT_SIZE         = 4        # Prompts sampled per tournament round
N_ELITES                = 8        # Top prompts always carried to next generation
N_CHILDREN              = 12       # Generated per generation (must equal 20 - N_ELITES)

# Prompt Generator
VARIATION_BATCH_SIZE    = 6        # Children per LLM call (2 calls × 6 = 12)
LLM_MAX_RETRIES         = 2
LLM_MODEL_NAME          = "<reasoning model identifier>"
MUTATION_CROSSOVER_RATIO = 0.5     # 50% of batches use crossover vs mutation

# Caching
ENABLE_EVAL_CACHE       = True

# Logging
LOG_DIR                 = "logs/"
CHECKPOINT_EVERY_N      = 2
```

---

## 14. File-Level Change Summary

```
project/
│
├── fitness_vector.py              [NEW]       FitnessVector dataclass
├── lineage_tracker.py             [NEW]       Full ancestry tree tracking
├── evaluation_cache.py            [NEW]       Hash-keyed evaluation memoization
├── prompt_generator.py            [NEW]       Batch semantic generation via reasoning LLM
├── tournament_selector.py         [NEW]       GA tournament selection
├── server_evolution_engine.py     [NEW]       Full server-side evolutionary orchestration
├── generation_logger.py           [NEW]       Snapshot serialization
├── output_manager.py              [NEW]       Three final output derivations
├── ablation_runner.py             [NEW]       LLM-only vs GA+LLM comparison
│
├── prompt_candidate.py            [MODIFY]    Add FitnessVector, lineage fields
├── population_manager.py          [MODIFY]    Rank by FitnessVector; integrate cache
├── federated_client.py            [MODIFY]    Remove evolution; keep evaluate_population only
├── federated_server.py            [MODIFY]    Orchestrate full evolutionary loop
├── run_experiment.py              [MODIFY]    Rewire main loop; ablation mode flag
├── config.py                      [MODIFY]    Add new configuration keys
│
├── model_loader.py                [UNCHANGED]
├── dataset_loader.py              [UNCHANGED]
├── evaluation_pipeline.py         [UNCHANGED]
├── prompt_scorer.py               [UNCHANGED]
├── seed_prompts.py                [UNCHANGED]
└── mutation_operators.py          [REMOVE]    Replaced by PromptGenerator
```

---

## 15. Non-Goals (Explicitly Deferred)

The following are **not implemented** in this paper:

- Differential Privacy
- Secure Aggregation
- Membership Inference defenses
- Flower (`flwr`) framework integration
- Post-hoc LLM synthesis of the deployment prompt (valid future extension if subsequently evaluated)

---

*End of Implementation Plan — FedGAPrompt v3 / HFPO (Revised)*
