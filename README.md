# FedGAPrompt: Federated Genetic Algorithm for Prompt Optimization

**Heterogeneous Federated Prompt Optimization (HFPO)** — A privacy-preserving framework that evolves system prompts across isolated medical institutions using genetic algorithms, without sharing patient data.

---

## What This Project Does

FedGAPrompt solves a key challenge in medical AI: **how to optimize prompts for clinical tasks when patient data cannot leave hospital servers**.

- **Federated**: Each "hospital" evaluates prompts on its own private dataset locally. No raw data, predictions, or model internals leave the hospital.
- **Genetic Algorithm**: Prompts evolve over generations via mutation, crossover, and elitism — guided by aggregated fitness scores from all hospitals.
- **Heterogeneous**: Each hospital uses a *different* medical dataset (MedQA, PubMedQA, MedMCQA), so prompts generalize across question distributions.
- **Adaptive Mutation**: Mutation strategies themselves evolve via meta-mutation, discovering better prompt-editing tactics over time.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a full HFPO evolutionary experiment
python -m experiments.run_evolution

# 3. Resume from latest checkpoint
python -m experiments.run_evolution --resume
```

**Outputs** appear under `results/hfpo_run/`:
```
results/hfpo_run/
├── best_prompts/best_prompt.txt      # Best prompt found
├── checkpoints/                      # Per-generation population checkpoints
├── snapshots/                        # Generation snapshots (CSV + JSON)
├── parent_child_diagnostics.jsonl    # Fitness delta & prediction agreement logs
└── mutation_statistics.json          # Mutation operator performance tracking
```

---

## Key Experiments

| Script | Purpose |
|--------|---------|
| `experiments/run_evolution.py` | **Main HFPO run** — federated GA across 3 hospitals |
| `experiments/run_baseline.py` | Centralized (non-federated) baseline comparison |
| `experiments/run_random_search.py` | Random search baseline |
| `experiments/run_centralized_ablation.py` | Ablation: centralized vs federated evaluation |
| `experiments/run_answering_engine.py` | Non-evolutionary answering engine baseline |

---

## Configuration

All tunable parameters live in `configs/config.py`:

### Model Selection
```python
DEFAULT_MODEL = "qwen"  # Options: qwen, phi-4, llama3-8b, mistral-7b, II-medical, meditron, oss
```

### Genetic Algorithm
```python
GA_POPULATION_SIZE = 20
GA_GENERATIONS = 20
GA_MUTATION_RATE = 0.35
GA_CROSSOVER_RATE = 0.65
GA_TOURNAMENT_SIZE = 3
```

### Adaptive Mutation (enabled by default)
```python
USE_ADAPTIVE_MUTATION = True
MUTATION_EVOLUTION_INTERVAL = 5   # Generations between meta-mutation
MUTATION_TOURNAMENT_SIZE = 3
MUTATION_ELITE_COUNT = 2
MIN_CHILDREN_FOR_RANKING = 10
```

### Evaluation
```python
EVALUATION_SUBSET_SIZE = 100  # Samples per hospital per prompt
RANDOM_SEED = 51
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EvolutionEngine                            │
│  (orchestrates generations: evaluate → select → generate → loop)│
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Hospital 1  │ │ Hospital 2  │ │ Hospital 3  │
    │  (MedQA)    │ │ (PubMedQA)  │ │ (MedMCQA)   │
    │  Evaluator  │ │  Evaluator  │ │  Evaluator  │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                  ┌─────────────────┐
                  │   Aggregator    │
                  │ (min/avg/max    │
                  │  across hospitals)│
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  FitnessVector  │
                  │  attached to    │
                  │  PromptCandidate│
                  └─────────────────┘
```

**Core Components** (`src/`):
| Module | Responsibility |
|--------|----------------|
| `core/` | Evolutionary data structures (`PromptCandidate`, `Population`, `FitnessVector`, `LineageTracker`, `EvaluationCache`) |
| `federated/` | `FederatedServer` (coordinator), `HospitalClient` (isolated evaluator), `Aggregator` |
| `evolution/` | `EvolutionEngine`, `PromptGenerator`, `TournamentSelector`, `Elitism`, mutation prompt evolution |
| `evaluation/` | `Evaluator` protocol, `QwenEvaluator`, metrics, parsers |
| `data/` | Medical dataset loaders (MedQA, PubMedQA, MedMCQA) |
| `llms/` | Model loading with 4-bit quantization (bitsandbytes) |
| `answering_engine/` | Centralized baseline (non-federated) |

---

## Directory Structure with Purposes

```
.
├── HFPO_results_tracker.md              / Tracks HFPO experiment results and metrics
├── _testresults/                        / Temporary test result storage
│   └── hfpo_run/                        / HFPO run test artifacts
│       ├── best_prompts/                / Best prompts from test runs
│       │   └── best_prompt.txt          / Best prompt text from test run
│       ├── checkpoints/                 / Test run checkpoints
│       ├── snapshots/                   / Test run generation snapshots
│       └── test/                        / Test output directory
├── configs/                             / Configuration management
│   └── config.py                        / Central config (models, GA params, quantization, generation settings)
├── experiments/                         / Experiment entry points and scripts
│   ├── audit_prompt_quality.py          / Audits prompt quality metrics
│   ├── run_answering_engine.py          / Runs centralized answering engine baseline
│   ├── run_baseline.py                  / Runs baseline comparison experiments
│   ├── run_centralized_ablation.py      / Runs centralized ablation study
│   ├── run_evolution.py                 / Main HFPO evolutionary optimization entry point
│   ├── run_model_test.py                / Tests model loading and inference
│   ├── run_random_search.py             / Random search baseline experiment
│   ├── test_prompt_generator.py         / Tests prompt generation pipeline
│   └── trace_generation_pipeline.py     / Traces prompt generation pipeline
├── outputs/                             / Experiment output directory
│   ├── Experiments/                     / Experiment-specific outputs
│   ├── output/                          / General output storage
│   └── output.txt                       / Consolidated output log
├── requirements.txt                     / Python dependencies
├── results/                             / Experiment results storage
│   ├── answering_engine/                / Answering engine results
│   │   └── report.txt                   / Answering engine evaluation report
│   ├── centralized_ablation/            / Centralized ablation study results
│   │   ├── best_prompts/                / Best prompts from ablation
│   │   ├── checkpoints/                 / Ablation checkpoints
│   │   └── snapshots/                   / Ablation generation snapshots
│   ├── hfpo_run/                        / Main HFPO run results
│   │   ├── best_prompts/                / Best prompts from HFPO run
│   │   │   └── best_prompt.txt          / Best prompt text
│   │   ├── checkpoints/                 / HFPO run checkpoints
│   │   ├── main.ipynb                   / Jupyter notebook for analysis
│   │   └── snapshots/                   / HFPO generation snapshots
│   └── prev/                            / Previous experiment results
│       └── contamination_baseline.md    / Contamination baseline documentation
├── run_diagnostics.py                   / Diagnostic script for system checks
├── setup.sh                             / Environment setup script
├── src/                                 / Source code root
│   ├── README.md                        / Source code documentation
│   ├── answering_engine/                / Centralized answering engine (non-federated baseline)
│   │   ├── dataset_loader.py            / Loads medical datasets for answering engine
│   │   ├── engine.py                    / Core answering engine orchestration
│   │   ├── export_manager.py            / Exports results and predictions
│   │   ├── metrics_calculator.py        / Calculates evaluation metrics
│   │   ├── models.py                    / Data models for answering engine
│   │   ├── prediction_extractor.py      / Extracts predictions from model outputs
│   │   ├── prompt_loader.py             / Loads prompts for answering engine
│   │   ├── report_generator.py          / Generates evaluation reports
│   │   └── response_generator.py        / Generates model responses
│   ├── core/                            / Core evolutionary data structures and utilities
│   │   ├── evaluation_cache.py          / Caches federated evaluation results
│   │   ├── evaluation_record.py         / Records per-hospital evaluation results
│   │   ├── fitness_vector.py            / Fitness vector with min/avg/max aggregation
│   │   ├── lineage_tracker.py           / Tracks prompt ancestry and lineage
│   │   ├── mutation_prompt_candidate.py / Mutation prompt candidate representation
│   │   ├── mutation_prompt_manager.py   / Manages adaptive mutation prompt population
│   │   ├── population.py                / Population container for GA generations
│   │   ├── prompt_candidate.py          / Core prompt candidate representation
│   │   └── seed_prompts.py              / Seed prompts for generation 0
│   ├── data/                            / Medical dataset loaders and preprocessing
│   │   ├── loader.py                    / Unified dataset loading interface
│   │   ├── medmcqa.py                   / MedMCQA dataset loader
│   │   ├── medqa.py                     / MedQA dataset loader
│   │   ├── preprocess.py                / Data preprocessing utilities
│   │   └── pubmedqa.py                  / PubMedQA dataset loader
│   ├── evaluation/                      / Federated evaluation components
│   │   ├── centralized_evaluator.py     / Centralized evaluation for baselines
│   │   ├── evaluator.py                 / Base evaluator interface
│   │   ├── metrics.py                   / Evaluation metrics computation
│   │   ├── parser.py                    / Model output parser
│   │   ├── qwen_evaluator.py            / Qwen-specific evaluator
│   │   └── scorer.py                    / Scoring utilities
│   ├── evolution/                       / Evolutionary algorithm orchestration
│   │   ├── elitism.py                   / Elite selection strategy
│   │   ├── evolution_engine.py          / Top-level evolution orchestration engine
│   │   ├── generation_snapshot.py       / Generation snapshot data structure
│   │   ├── mutation_prompt_logger.py    / Logs mutation prompt history
│   │   ├── output_manager.py            / Manages experiment outputs and checkpoints
│   │   ├── prompt_generator/            / Prompt generation pipeline
│   │   │   ├── candidate_parser.py      / Parses LLM-generated candidates
│   │   │   ├── cleaner.py               / Cleans raw LLM output
│   │   │   ├── generator.py             / Orchestrates prompt generation pipeline
│   │   │   ├── interfaces.py            / LLM interface protocols
│   │   │   ├── models.py                / Generation request/result data models
│   │   │   ├── mutation_statistics.py   / Tracks mutation operator statistics
│   │   │   ├── prompts/                 / Mutation prompt templates and strategies
│   │   │   │   ├── mutation/            / Mutation operator implementations
│   │   │   │   │   ├── aggressive.py    / Aggressive mutation strategy
│   │   │   │   │   ├── differential.py  / Differential mutation strategy
│   │   │   │   │   ├── elimination.py   / Elimination mutation strategy
│   │   │   │   │   ├── evidence.py      / Evidence-based mutation strategy
│   │   │   │   │   ├── guideline.py     / Guideline mutation strategy
│   │   │   │   │   ├── meta_mutation.py / Meta-mutation strategy
│   │   │   │   │   ├── probability.py   / Probability mutation strategy
│   │   │   │   │   ├── reasoning.py     / Reasoning mutation strategy
│   │   │   │   │   └── role.py          / Role-based mutation strategy
│   │   │   │   └── prompt_utils/        / Prompt utility functions
│   │   │   │       ├── crossover.py     / Crossover prompt utilities
│   │   │   │       └── mutation.py      / Mutation prompt utilities
│   │   │   ├── similarity_selector.py   / Selects diverse candidates
│   │   │   ├── templates.py             / Prompt template builder
│   │   │   ├── thinking_directions.py   / Reasoning direction templates
│   │   │   └── validator.py             / Validates generated prompts
│   │   ├── prompt_logger.py             / Logs generated prompts
│   │   ├── similarity.py                / Prompt similarity computation
│   │   └── tournament_selector.py       / Tournament selection for parents
│   ├── federated/                       / Federated learning orchestration
│   │   ├── aggregator.py                / Aggregates hospital evaluation results
│   │   ├── federated_server.py          / Central federated evaluation coordinator
│   │   └── hospital_client.py           / Individual hospital client with private data
│   ├── llms/                            / LLM loading and inference
│   │   ├── llm.py                       / LLM interface and base classes
│   │   ├── loader.py                    / Model loading with quantization
│   │   └── test.py                      / Model loading tests
│   └── prompts/                         / Prompt building utilities
│       ├── baseline.py                  / Baseline prompt templates
│       └── builder.py                   / Prompt builder utilities
└── tests/                               / Unit tests
    └── test_parser.py                   / Parser unit tests
```

---

## Output Files Explained

| File | Description |
|------|-------------|
| `results/hfpo_run/best_prompts/best_prompt.txt` | Highest-fitness prompt discovered |
| `results/hfpo_run/snapshots/gen_*.json` | Full generation state (prompts, fitness, stats) |
| `results/hfpo_run/snapshots/gen_*.csv` | Tabular summary for spreadsheet analysis |
| `results/hfpo_run/checkpoints/gen_*.pkl` | Pickled `Population` for `--resume` |
| `results/hfpo_run/parent_child_diagnostics.jsonl` | Per-child fitness deltas + prediction agreement vs parents |
| `results/hfpo_run/mutation_statistics.json` | Which mutation operators improve fitness most |

---

## Requirements

- Python 3.10+
- PyTorch with CUDA
- `bitsandbytes` for 4-bit quantization
- `transformers`, `accelerate`, `peft`
- Medical datasets: `datasets` (Hugging Face) for MedQA, PubMedQA, MedMCQA

Install:
```bash
pip install -r requirements.txt
```

---

## Citation

If you use FedGAPrompt in research, please cite:

```bibtex
@misc{fedgaprompt2024,
  title={FedGAPrompt: Federated Genetic Algorithm for Prompt Optimization},
  author={...},
  year={2024}
}
```

---

## License

MIT License — see `LICENSE` for details.