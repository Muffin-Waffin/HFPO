import os
import json
import csv
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    print("=" * 80)
    print("RUNNING CORRECTED HFPO EVOLUTION DIAGNOSTICS & CONVERGENCE ANALYSIS")
    print("=" * 80)

    # 1. Paths
    results_dir = Path("results/hfpo_run")
    checkpoints_dir = results_dir / "checkpoints"
    snapshots_dir = results_dir / "snapshots"
    generated_prompts_file = results_dir / "generated_prompts.jsonl"
    diagnostics_file = results_dir / "parent_child_diagnostics.jsonl"

    # 2. Build fitness lookup map from parent_child_diagnostics.jsonl
    print("Building fitness lookup map from diagnostics log...")
    fitness_lookup = {}
    
    if diagnostics_file.exists():
        with open(diagnostics_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                child_id = record.get("child_id")
                child_fitness = record.get("child_fitness")
                if child_id and child_fitness:
                    fitness_lookup[child_id] = child_fitness
    print(f"Loaded fitness data for {len(fitness_lookup)} candidates from diagnostics log.")

    # 3. Load all generated prompts for lineage mapping (metadata lookup)
    print("Loading generated prompts metadata...")
    prompt_metadata = {}
    if generated_prompts_file.exists():
        with open(generated_prompts_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                cand_id = record.get("candidate_id")
                if cand_id:
                    prompt_metadata[cand_id] = {
                        "origin": record.get("origin"),
                        "mutation_operator": record.get("metadata", {}).get("mutation_operator"),
                        "generation_time_ms": record.get("metadata", {}).get("generation_time_ms"),
                        "attempts": record.get("metadata", {}).get("attempts", 1),
                    }

    # 4. Find and load all checkpoints
    checkpoint_files = sorted(checkpoints_dir.glob("population_checkpoint_*.json"))
    if not checkpoint_files:
        print("No population checkpoints found.")
        return
    print(f"Found {len(checkpoint_files)} checkpoint files.")

    # We need to build a global database of all candidates across checkpoints
    all_candidates = {}
    generations_data = {}

    for cp_file in checkpoint_files:
        with open(cp_file, "r") as f:
            data = json.load(f)
        gen = data["generation"]
        candidates = data["candidates"]
        
        generations_data[gen] = candidates
        for cand in candidates:
            all_candidates[cand["id"]] = cand
            # If the candidate has fitness in the checkpoint, add it to lookup
            if cand.get("fitness"):
                fitness_lookup[cand["id"]] = cand["fitness"]

    # Sort generations
    sorted_gens = sorted(generations_data.keys())
    print(f"Generations found: {sorted_gens}")

    # Load SentenceTransformer for semantic similarity
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    try:
        similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✓ SentenceTransformer loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load sentence-transformers: {e}. Semantic similarity will be skipped.")
        similarity_model = None

    # Track evaluated prompts for cache hit analysis
    evaluated_prompt_texts = set()
    
    rows = []

    # Process Generation 0 from snapshot
    snapshot_0 = snapshots_dir / "generation_000.json"
    if snapshot_0.exists():
        with open(snapshot_0, "r") as f:
            d0 = json.load(f)
        
        # Estimate variance or set to None for generation 0 since we do not have all 20 individual scores
        # We can extract text of the 2 elites from checkpoint 1
        d1_cands = generations_data.get(1, [])
        elites_texts = [c["text"] for c in d1_cands if c["generation"] == 0]
        for t in elites_texts:
            evaluated_prompt_texts.add(t.strip())

        # For mean pairwise similarity, we can load seed prompts from SEED_PROMPTS
        mean_sim_0 = 0.0
        if similarity_model:
            from src.core.seed_prompts import SEED_PROMPTS
            embeddings = similarity_model.encode(SEED_PROMPTS, normalize_embeddings=True)
            sim_matrix = embeddings @ embeddings.T
            off_diag_sims = []
            for i in range(len(SEED_PROMPTS)):
                for j in range(i + 1, len(SEED_PROMPTS)):
                    off_diag_sims.append(sim_matrix[i, j])
            mean_sim_0 = float(np.mean(off_diag_sims))

        rows.append({
            "generation": 0,
            "mean_fitness": d0["average_score"],
            "best_fitness": d0["best_score"],
            "worst_fitness": d0["worst_score"],
            "fitness_variance": 0.0,  # Indeterminate without individual scores
            "mean_pairwise_similarity": mean_sim_0,
            "elitism_count": 0,
            "elitism_fraction": 0.0,
            "cache_hit_count": 0,
            "cache_hit_rate": 0.0,
            "seed_count": 20,
            "crossover_count": 0,
            "mutation_count": 0,
            "active_strategies_count": 0,
            "strategy_distribution": "",
        })

    # Process each generation G >= 1
    for gen in sorted_gens:
        print(f"Processing Generation {gen}...")
        population = generations_data[gen]
        pop_size = len(population)

        # Compute fitness stats
        fitness_values = []
        for cand in population:
            cand_id = cand["id"]
            fitness = cand.get("fitness") or fitness_lookup.get(cand_id)
            if fitness:
                avg_score = sum(fitness.values()) / len(fitness)
                fitness_values.append(avg_score)
                # Keep lookup updated
                fitness_lookup[cand_id] = fitness
            else:
                print(f"  Warning: No fitness found for candidate {cand_id[:8]} in generation {gen}")
                fitness_values.append(0.0)

        mean_fit = np.mean(fitness_values) if fitness_values else 0.0
        best_fit = np.max(fitness_values) if fitness_values else 0.0
        worst_fit = np.min(fitness_values) if fitness_values else 0.0
        var_fit = np.var(fitness_values) if fitness_values else 0.0

        # Compute pairwise embedding similarity
        mean_sim = 0.0
        if similarity_model and population:
            texts = [cand["text"] for cand in population]
            embeddings = similarity_model.encode(texts, normalize_embeddings=True)
            sim_matrix = embeddings @ embeddings.T
            n = len(texts)
            if n > 1:
                off_diag_sims = []
                for i in range(n):
                    for j in range(i + 1, n):
                        off_diag_sims.append(sim_matrix[i, j])
                mean_sim = float(np.mean(off_diag_sims))
            else:
                mean_sim = 1.0

        # Cache hit rate analysis
        cache_hits = 0
        for cand in population:
            text = cand["text"].strip()
            if text in evaluated_prompt_texts:
                cache_hits += 1
            else:
                evaluated_prompt_texts.add(text)
        
        cache_hit_rate = cache_hits / pop_size if pop_size > 0 else 0.0

        # Elitism analysis
        elitism_count = 0
        if gen > 0:
            for cand in population:
                if cand["generation"] < gen:
                    elitism_count += 1
        
        elitism_fraction = elitism_count / pop_size if pop_size > 0 else 0.0

        # Candidate origin and strategy diversity analysis
        crossover_count = 0
        mutation_count = 0
        seed_count = 0
        strategy_counts = {}

        for cand in population:
            cand_id = cand["id"]
            origin = cand.get("origin", "unknown")
            
            meta = prompt_metadata.get(cand_id, {})
            operator = meta.get("mutation_operator") or cand.get("metadata", {}).get("mutation_operator")
            
            if origin == "unknown":
                origin = meta.get("origin") or "seed" if cand["generation"] == 0 else "unknown"

            if origin == "seed" or cand["generation"] == 0:
                seed_count += 1
            elif origin == "crossover" or meta.get("origin") == "crossover":
                crossover_count += 1
                strategy_counts["crossover"] = strategy_counts.get("crossover", 0) + 1
            elif origin == "mutation" or meta.get("origin") == "mutation":
                mutation_count += 1
                op_name = operator or "mutation_unknown"
                strategy_counts[op_name] = strategy_counts.get(op_name, 0) + 1
            else:
                orig_cand = all_candidates.get(cand_id, cand)
                orig_meta = prompt_metadata.get(cand_id, {})
                orig_op = orig_meta.get("mutation_operator") or orig_cand.get("metadata", {}).get("mutation_operator")
                orig_origin = orig_cand.get("origin", "seed") if orig_cand["generation"] == 0 else orig_meta.get("origin", "unknown")
                
                if orig_origin == "seed" or orig_cand["generation"] == 0:
                    seed_count += 1
                elif orig_origin == "crossover":
                    crossover_count += 1
                    strategy_counts["crossover"] = strategy_counts.get("crossover", 0) + 1
                else:
                    mutation_count += 1
                    op_name = orig_op or "mutation_unknown"
                    strategy_counts[op_name] = strategy_counts.get(op_name, 0) + 1

        active_strategies_count = len(strategy_counts)
        strategy_dist_str = "; ".join([f"{k}:{v}" for k, v in sorted(strategy_counts.items())])

        row = {
            "generation": gen,
            "mean_fitness": float(mean_fit),
            "best_fitness": float(best_fit),
            "worst_fitness": float(worst_fit),
            "fitness_variance": float(var_fit),
            "mean_pairwise_similarity": float(mean_sim),
            "elitism_count": elitism_count,
            "elitism_fraction": float(elitism_fraction),
            "cache_hit_count": cache_hits,
            "cache_hit_rate": float(cache_hit_rate),
            "seed_count": seed_count,
            "crossover_count": crossover_count,
            "mutation_count": mutation_count,
            "active_strategies_count": active_strategies_count,
            "strategy_distribution": strategy_dist_str,
        }
        rows.append(row)

    # 5. Save results to CSV and JSON
    csv_file = results_dir / "convergence_analysis.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_file = results_dir / "convergence_analysis.json"
    with open(json_file, "w") as f:
        json.dump(rows, f, indent=4)

    print("=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print(f"CSV report saved to: {csv_file}")
    print(f"JSON report saved to: {json_file}")
    print("=" * 80)

    # Print a summary table
    print(f"{'Gen':<4} | {'Mean Fit':<8} | {'Best Fit':<8} | {'Worst Fit':<8} | {'Variance':<8} | {'Similarity':<10} | {'Cache Hit %':<11} | {'Strategies':<10}")
    print("-" * 95)
    for r in rows:
        print(f"{r['generation']:<4} | {r['mean_fitness']:<8.4f} | {r['best_fitness']:<8.4f} | {r['worst_fitness']:<8.4f} | {r['fitness_variance']:<8.6f} | {r['mean_pairwise_similarity']:<10.4f} | {r['cache_hit_rate']*100:<10.1f}% | {r['active_strategies_count']:<10}")

if __name__ == "__main__":
    main()
