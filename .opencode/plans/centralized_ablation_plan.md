# Centralized Evaluation Ablation Experiment Plan

## Overview
Build a centralized evaluation backend (`CentralizedEvaluator`) and experiment script (`run_centralized_ablation.py`) to compare against the federated HFPO run for Experiment 4.

---

## Component 1: CentralizedEvaluator (`src/evaluation/centralized_evaluator.py`)

### Purpose
Drop-in replacement for `FederatedServer` with identical `evaluate_population(population) -> population` interface, but using pooled centralized evaluation instead of federated broadcast/aggregate.

### Key Design Decisions

1. **Pooled Dataset Construction**
   - Load all 3 datasets (medqa, pubmedqa, medmcqa) with same `EVALUATION_SUBSET_SIZE=100` and `RANDOM_SEED=51` as federated run
   - Use `random.sample` with identical seed to select same 100 indices per dataset
   - Concatenate into single 300-sample evaluation set
   - Same underlying data as federated per-hospital subsets — only combination differs

2. **Fitness Computation: Micro-Average (Centralized) vs Macro-Average (Federated)**
   - Federated: `FitnessVector.average()` = mean of 3 per-hospital scores (macro-average)
   - Centralized: `total_correct / 300` across pooled set (micro-average)
   - **This is the intentional experimental difference** — documented in class docstring

3. **FitnessVector Wrapper for Compatibility**
   - `PromptCandidate.fitness` expects `FitnessVector` (per `src/core/prompt_candidate.py`)
   - `EvolutionEngine` uses `candidate.fitness.average()` for selection/elitism
   - Wrap single centralized score in `FitnessVector({"centralized": score})`
   - `.average()` returns the same scalar → zero downstream changes needed

4. **Evaluation Pipeline Reuse**
   - Reuse `QwenEvaluator.score_sample_details()` for per-sample scoring
   - Same model, tokenizer, parser, metrics as federated hospitals

5. **Interface Parity with FederatedServer**
   - `evaluate_population(population) -> population`
   - `cache_statistics()`
   - `summary()`
   - `__str__()`
   - Optional: `evaluation_cache` and `lineage_tracker` integration for caching/resume

### Constructor Signature
```python
def __init__(
    self,
    model: Any,
    tokenizer: Any,
    datasets: List[Any],  # pre-loaded pooled dataset
    evaluation_cache: EvaluationCache,
    lineage_tracker: LineageTracker,
    model_name: str,
    dataset_name: str,  # e.g., "medqa+pubmedqa+medmcqa"
    evaluation_version: str,
)
```

### evaluate_population() Logic
```python
def evaluate_population(self, population: List[PromptCandidate]) -> List[PromptCandidate]:
    # Check cache for each prompt
    # For uncached: evaluate on pooled dataset using QwenEvaluator
    # Compute micro-average accuracy = total_correct / 300
    # Wrap in FitnessVector({"centralized": accuracy})
    # Store in cache
    # Register with lineage_tracker
    return population
```

---

## Component 2: run_centralized_ablation.py (`experiments/run_centralized_ablation.py`)

### Structure (mirrors run_evolution.py)

1. **Constants** (same as federated run):
   - `GA_POPULATION_SIZE = 20` (from config)
   - `GA_GENERATIONS = 10` (plateau cap, not config's 20)
   - `ELITE_COUNT = 2`
   - `RANDOM_SEED = 51` (config.RANDOM_SEED)
   - `EVALUATION_SUBSET_SIZE = 100` (config)
   - `DATASET_SPLIT = "train"`
   - Same `SEED_PROMPTS`, `SEED_MUTATION_STRATEGIES`
   - Same `TASK_DESCRIPTION`

2. **Argument Parser**:
   - `--generations N` / `-g N` — override generation count (for smoke test)
   - `--limit N` — alias for --generations

3. **Main Flow**:
   - Load model/tokenizer (same as federated)
   - Load 3 datasets with same subset sampling (seed=51, size=100 each)
   - Build pooled dataset = concat all 3 subsets
   - Build `CentralizedEvaluator` with pooled dataset
   - Build same dependency graph: `PromptGenerator`, `TournamentSelector`, `Elitism`, `OutputManager`, `MutationPromptManager`
   - **Critical**: Use same `TOURNAMENT_RANDOM_SEED = 42` and `RANDOM_SEED = 51` for evolutionary RNG
   - Build seed population from `SEED_PROMPTS`
   - Construct `EvolutionEngine` with `CentralizedEvaluator` instead of `FederatedServer`
   - Run evolution

4. **Logging/Output**:
   - Console: Print best fitness per generation (same format as run_evolution.py)
   - CSV: `results/centralized_ablation_results.csv` with columns:
     - `generation`, `best_fitness`, `best_prompt_text`
   - Final per-dataset breakdown: evaluate best prompt on each of 3 original 100-sample subsets
   - Output directory: `results/centralized_ablation/`

5. **Resume/Checkpoint Support** (mirror federated):
   - Checkpoint loading from `results/centralized_ablation/checkpoints/`
   - Lineage tracker registration with `allow_missing_parents=True`

---

## Component 3: Integration Points

### EvolutionEngine Compatibility
- `EvolutionEngine` expects `federated_server` with `evaluate_population()`
- `CentralizedEvaluator` provides identical method signature
- `fitness.average()` works identically → zero changes to `EvolutionEngine`
- `Elitism`, `TournamentSelector` use `.average()` → works unchanged

### OutputManager
- Use `OutputManager(output_directory="results/centralized_ablation")`
- Snapshots, best prompts, checkpoints all work identically

---

## Smoke Test Plan

1. Run with `--generations 2`
2. Verify console output shows:
   - Generation 1 best fitness
   - Generation 2 best fitness
   - No errors
3. Verify `results/centralized_ablation_results.csv` exists with 2 rows
4. Verify per-dataset breakdown printed at end

---

## Full Run Plan

1. Run with default 10 generations (or `--generations 10`)
2. Compare `centralized_ablation_results.csv` vs federated run's fitness curve
3. Compare final per-dataset breakdowns

---

## File Changes Summary

| File | Action |
|------|--------|
| `src/evaluation/centralized_evaluator.py` | CREATE new |
| `src/evaluation/__init__.py` | EDIT: export CentralizedEvaluator |
| `experiments/run_centralized_ablation.py` | CREATE new |
| `results/centralized_ablation/` | AUTO-CREATED by OutputManager |
| `results/centralized_ablation_results.csv` | AUTO-CREATED by experiment script |

---

## Key Implementation Details to Verify

1. **Dataset sampling reproducibility**: Use `random.Random(RANDOM_SEED).sample(...)` not global `random.sample` to avoid cross-contamination with evolutionary RNG
2. **Evolution RNG isolation**: `EvolutionEngine` uses its own `self._operator_rng = random.Random()` — ensure we don't share RNG with dataset sampling
3. **FitnessVector wrapper**: `FitnessVector({"centralized": score})` — single key, `.average()` returns score
4. **Cache key**: Use `dataset_name="medqa+pubmedqa+medmcqa"` so cache is separate from federated run
5. **QwenEvaluator reuse**: Use `scorer.score_sample()` via `QwenEvaluator` — same pipeline as hospitals