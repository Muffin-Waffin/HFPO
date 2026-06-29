### Keep

* ✅ Federated evaluation
* ✅ GA (Tournament Selection + Elitism)
* ✅ LLM Prompt Generator
* ✅ FitnessVector
* ✅ Evaluation Cache
* ✅ Lineage Tracker
* ✅ Ablation (LLM vs GA+LLM)
* ✅ Local Best + Federated Best prompts

### Remove

* ❌ Deployment Prompt (it's identical to Federated Best)
* ❌ "Outperform both parents" wording in LLM prompts (replace with "attempt to improve")
* ❌ Top-k selection as the main mechanism (keep only elitism)

### Add

* ✅ EvaluationRecord logging
* ✅ Population diversity metric
* ✅ Versioned evaluation cache
* ✅ True non-IID hospital partitions
* ✅ Better experiment logging

---

# Final Architecture (v4)

I'd rename the project architecture to:

> **FedGAPrompt v4: Federated LLM-Guided Evolutionary Prompt Optimization**

The paper becomes much cleaner.

---

# Final Folder Structure

```
FedGAPrompt/
│
├── configs/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_ga.py
│   ├── run_federated.py
│   ├── run_ablation.py
│   └── run_hfpo.py
│
├── logs/
│
├── results/
│
├── src/
│   │
│   ├── data/
│   ├── models/
│   ├── prompts/
│   ├── evaluation/
│   │
│   ├── evolution/
│   │   ├── fitness_vector.py
│   │   ├── prompt_candidate.py
│   │   ├── tournament_selector.py
│   │   ├── prompt_generator.py
│   │   ├── lineage_tracker.py
│   │   ├── evaluation_cache.py
│   │   ├── diversity.py
│   │   └── evolution_engine.py
│   │
│   ├── federated/
│   │   ├── client.py
│   │   ├── server.py
│   │   ├── aggregator.py
│   │   └── partitioner.py
│   │
│   └── utils/
│
└── tests/
```

---

# Final Algorithm

```
Initialize

↓

20 Seed Prompts

↓

Broadcast to Hospitals

↓

Evaluate Locally

↓

Return Scores

↓

Fitness Vector

↓

Tournament Selection

↓

Elitism

↓

Prompt Generator (Reasoning LLM)

↓

12 New Prompts

↓

Evaluation Cache

↓

Next Generation

↓

Repeat
```

---

# Population

Always

```
20 prompts
```

Every generation.

```
8 elites

+

12 generated
```

No exceptions.

---

# Hospitals

Hospitals perform

ONLY

```
Receive prompts

↓

Evaluate

↓

Return scores
```

Nothing else.

---

# Server

The server owns

```
Selection

Generation

Caching

Logging

Aggregation

History
```

Everything evolutionary stays on the server.

---

# Fitness

Each prompt stores

```
Hospital A

Hospital B

Hospital C

Average

Minimum

Variance
```

No scalar fitness.

---

# Parent Selection

Tournament

```
Population

↓

Random sample of 4

↓

Winner

↓

Repeat
```

Elites survive.

Tournament chooses parents.

---

# Prompt Generator

Input

```
Task

Generation Number

Current Best Prompt

Parent Prompt(s)

Fitness Vector

Weakest Hospital

Strongest Hospital

Aggregate Failure Metadata (optional)
```

Output

```
JSON

[
prompt1,
prompt2,
prompt3,
...
]
```

Never output explanations.

---

# Cache

Key

```
SHA256(

Prompt

Model

Dataset Version

Evaluation Version

)
```

This avoids stale evaluations.

---

# Logging

Every generation log

```
Population

Fitness Vectors

Tournament Winners

Children

Lineage

Diversity

Evaluation Records
```

---

# Diversity Metric

Compute

```
Sentence Embedding

↓

Pairwise Cosine Similarity

↓

Population Diversity
```

This becomes one of the paper's figures.

---

# Final Outputs

Instead of three prompts

Produce only

## Local Best

One prompt per hospital.

```
Hospital A

Hospital B

Hospital C
```

---

## Federated Best

Highest average

```
Across every hospital

Across every generation
```

This is also your deployment prompt.

No additional LLM synthesis.

---

# Experiments

These become your paper.

## Experiment 1

Static Prompt

---

## Experiment 2

LLM Only

```
20 prompts

↓

LLM

↓

Evaluate
```

One iteration.

---

## Experiment 3

GA Only

Use your original handcrafted mutation/crossover (or disable LLM generation) to measure the contribution of the LLM itself.

---

## Experiment 4

GA + LLM

Single hospital.

---

## Experiment 5

Federated GA + LLM

Three hospitals.

This is the main experiment.

---

## Experiment 6

IID

vs

Non-IID

partitioning.

---

# Metrics

Report

* Accuracy
* Per-hospital accuracy
* Average fitness
* Worst-case fitness
* Fitness variance
* Population diversity
* Number of unique prompts generated
* Number of cache hits
* Runtime per generation

---

# Figures

Include:

1. Overall HFPO architecture.
2. One-generation workflow.
3. Evolutionary lineage tree.
4. Accuracy vs generation.
5. Diversity vs generation.
6. Per-hospital learning curves.
7. Ablation comparison.

---

# Milestones

1. Foundation (FitnessVector, PromptCandidate, Cache, Lineage).
2. Hospital evaluator.
3. Server aggregation.
4. Tournament + Elitism.
5. Prompt Generator.
6. One-generation HFPO.
7. Multi-generation HFPO.
8. Logging.
9. Ablation experiments.
10. Paper.

---