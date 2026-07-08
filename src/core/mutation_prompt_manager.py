"""Mutation prompt manager for adaptive mutation population."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any
import statistics
import uuid
from pathlib import Path

from src.core.mutation_prompt_candidate import MutationPromptCandidate
from src.core.prompt_candidate import PromptCandidate
from src.evolution.prompt_generator.interfaces import ReasoningLLM


@dataclass(slots=True)
class MutationRankingStrategy:
    """Configurable ranking strategy for mutation prompt fitness."""
    
    # Weights for different metrics
    weight_avg_improvement: float = 0.5
    weight_success_rate: float = 0.3
    weight_median_improvement: float = 0.2
    
    def compute_fitness(self, candidate: MutationPromptCandidate) -> float:
        """Compute fitness based on configurable weights."""
        if candidate.children_generated == 0:
            return 0.0
        
        avg_imp = candidate.average_improvement
        median_imp = candidate.median_improvement
        success = candidate.success_rate
        
        fitness = (
            self.weight_avg_improvement * avg_imp +
            self.weight_success_rate * success +
            self.weight_median_improvement * median_imp
        )
        return fitness


class MutationPromptManager:
    """Manages a population of mutation strategies.
    
    Handles selection, statistics tracking, and periodic evolution
    of mutation strategies. Lightweight - not a full GA engine.
    """
    
    def __init__(
        self,
        candidates: list[MutationPromptCandidate],
        tournament_size: int = 3,
        elite_count: int = 2,
        min_children_before_evolution: int = 10,
        ranking_strategy: MutationRankingStrategy | None = None,
        random_seed: int | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("MutationPromptManager requires at least one candidate.")
        
        self._candidates: dict[str, MutationPromptCandidate] = {
            c.id: c for c in candidates
        }
        self._generation = 0
        self._tournament_size = tournament_size
        self._elite_count = elite_count
        self._min_children = min_children_before_evolution
        self._ranking_strategy = ranking_strategy or MutationRankingStrategy()
        self._rng = Random(random_seed)
        
    @property
    def candidates(self) -> list[MutationPromptCandidate]:
        return list(self._candidates.values())
    
    @property
    def generation(self) -> int:
        return self._generation
    
    def select(self) -> MutationPromptCandidate:
        """Select a mutation strategy via tournament selection.
        
        Uses custom scoring based on fitness computed by ranking strategy.
        """
        # Ensure all candidates have fitness computed
        for c in self._candidates.values():
            if c.fitness is None and c.children_generated > 0:
                c.fitness = self._ranking_strategy.compute_fitness(c)
        
        # Use fitness for tournament scoring
        candidates_list = list(self._candidates.values())
        
        # If no one has fitness yet, pick randomly
        if all(c.fitness is None for c in candidates_list):
            return self._rng.choice(candidates_list)
        
        # Run tournament
        contestant_ids = self._rng.sample(
            list(self._candidates.keys()),
            min(self._tournament_size, len(self._candidates))
        )
        contestants = [self._candidates[cid] for cid in contestant_ids]
        
        # Score by fitness (or 0 if None)
        def score(c: MutationPromptCandidate) -> float:
            return c.fitness if c.fitness is not None else 0.0
        
        winner = max(contestants, key=score)
        winner.times_selected += 1
        return winner
    
    def record(
        self,
        mutation_prompt: MutationPromptCandidate,
        parent: PromptCandidate,
        child: PromptCandidate,
    ) -> None:
        """Record the outcome of using a mutation strategy."""
        if mutation_prompt.id not in self._candidates:
            raise ValueError(f"Mutation prompt {mutation_prompt.id} not in manager.")
        
        if parent.fitness is None or child.fitness is None:
            raise ValueError("Both parent and child must have fitness.")
        
        parent_avg = parent.fitness.average()
        child_avg = child.fitness.average()
        
        mutation_prompt.record_child_result(
            parent_id=parent.id,
            child_id=child.id,
            parent_fitness=parent_avg,
            child_fitness=child_avg,
            generation=self._generation,
        )
        
        # Recompute fitness
        mutation_prompt.fitness = self._ranking_strategy.compute_fitness(mutation_prompt)
    
    def increment_generation(self) -> None:
        """Call at the end of each task prompt generation."""
        self._generation += 1
        for c in self._candidates.values():
            c.increment_age()
    
    def should_evolve(self) -> bool:
        """Check if enough data collected to evolve mutation strategies."""
        if self._generation == 0:
            return False
        
        # All candidates must have minimum children
        if not all(c.children_generated >= self._min_children for c in self._candidates.values()):
            return False
        
        return True
    
    def evolve(
        self,
        llm: ReasoningLLM,
        task_description: str,
        template_builder=None,
        prompt_generator=None,
        temperature: float = 0.7,
    ) -> list[MutationPromptCandidate]:
        """Evolve mutation strategies using meta-mutation.
        
        Calls the LLM directly with the meta-mutation template to
        generate 5 candidate strategies, then selects the most novel
        one via similarity comparison.
        
        1. Rank by fitness, keep elites
        2. For each non-elite, generate 5 candidates via LLM and pick the best
        3. Return new population
        """
        from src.evolution.prompt_generator.candidate_parser import CandidateParser
        from src.evolution.prompt_generator.similarity_selector import PromptSimilaritySelector
        from src.evolution.prompt_generator.cleaner import PromptCleaner
        from src.evolution.prompt_generator.prompts.mutation.meta_mutation import META_MUTATION_TEMPLATE

        candidate_parser = CandidateParser()
        similarity_selector = PromptSimilaritySelector()
        cleaner = PromptCleaner()

        # Rank by fitness
        ranked = sorted(
            self._candidates.values(),
            key=lambda c: c.fitness if c.fitness is not None else 0.0,
            reverse=True,
        )
        
        elites = ranked[:self._elite_count]
        to_mutate = ranked[self._elite_count:]
        
        new_candidates = list(elites)  # Keep elites unchanged
        
        # Build context for meta-mutation
        best_candidate = ranked[0] if ranked else None
        
        max_retries = 5

        for parent in to_mutate:
            # Build performance summary
            performance_summary = self._build_performance_summary(
                parent, best_candidate
            )

            # Build meta-mutation instruction
            instruction = META_MUTATION_TEMPLATE.format(
                task_description=task_description,
                current_strategy=parent.strategy,
                performance_summary=performance_summary,
            )

            generated = False
            for attempt in range(max_retries):
                try:
                    print(f"  Meta-mutating strategy {parent.id[:8]}... (attempt {attempt + 1}/{max_retries})")

                    # Call LLM directly
                    raw_output = llm.generate(
                        prompt=instruction,
                        temperature=temperature,
                    )

                    # Parse candidates (resilient: returns 1-5 candidates)
                    candidates = candidate_parser.parse(raw_output)

                    # Select the most novel candidate
                    selected_strategy = similarity_selector.select(
                        parent_prompt=parent.strategy,
                        candidates=candidates,
                    )

                    # Clean the selected strategy
                    clean_strategy = cleaner.clean(selected_strategy)

                    if not clean_strategy.strip():
                        print(f"    Empty strategy after cleaning, retrying...")
                        continue

                    # Create new mutation candidate
                    # Derive name from parent, suffixed with generation
                    new_name = f"{parent.name} v{self._generation + 1}"
                    new_candidate = MutationPromptCandidate(
                        id=str(uuid.uuid4()),
                        name=new_name,
                        strategy=clean_strategy,
                        generation=self._generation + 1,
                        parent_ids=[parent.id],
                        age=0,
                        metadata={"origin": "meta_mutation", "parent_fitness": parent.fitness},
                    )
                    new_candidates.append(new_candidate)
                    generated = True
                    break

                except Exception as e:
                    print(f"    Attempt {attempt + 1}/{max_retries} failed: {e}")

            if not generated:
                # Keep the parent if all retries failed
                print(f"  WARNING: Failed to meta-mutate strategy {parent.id[:8]}, keeping parent.")
                new_candidates.append(parent)
        
        # Replace population
        self._candidates = {c.id: c for c in new_candidates}
        return new_candidates

    def _build_performance_summary(
        self,
        parent: MutationPromptCandidate,
        best_candidate: MutationPromptCandidate | None,
    ) -> str:
        """Build a performance summary string for the meta-mutation template."""
        perf_lines = [
            f"Current Strategy: {parent.strategy}",
            f"Generation: {parent.generation}",
            f"Age: {parent.age}",
            f"Times Selected: {parent.times_selected}",
            f"Children Generated: {parent.children_generated}",
            f"Average Improvement: {parent.average_improvement:.4f}",
            f"Median Improvement: {parent.median_improvement:.4f}",
            f"Best Improvement: {parent.best_improvement:.4f}",
            f"Worst Improvement: {parent.worst_improvement:.4f}",
            f"Success Rate: {parent.success_rate:.2%}",
            f"Fitness: {parent.fitness:.4f}" if parent.fitness is not None else "Fitness: N/A",
        ]
        
        # Add top successful children
        if parent.top_successful_children:
            perf_lines.append("\nTop Successful Children:")
            for i, child in enumerate(parent.top_successful_children[:3], 1):
                perf_lines.append(
                    f"  {i}. Parent: {child['parent_fitness']:.3f} -> "
                    f"Child: {child['child_fitness']:.3f} "
                    f"(+{child['improvement']:.3f})"
                )
        
        # Add top failed children
        if parent.top_failed_children:
            perf_lines.append("\nTop Failed Children:")
            for i, child in enumerate(parent.top_failed_children[:3], 1):
                perf_lines.append(
                    f"  {i}. Parent: {child['parent_fitness']:.3f} -> "
                    f"Child: {child['child_fitness']:.3f} "
                    f"({child['improvement']:.3f})"
                )
        
        # Add best strategy context
        if best_candidate and best_candidate.id != parent.id:
            best_fitness_str = (
                f"{best_candidate.fitness:.4f}"
                if best_candidate.fitness is not None
                else "N/A"
            )
            perf_lines.append(f"\nBest Strategy in Population (Fitness: {best_fitness_str}):")
            perf_lines.append(f"  Strategy: {best_candidate.strategy}")
            perf_lines.append(f"  Avg Improvement: {best_candidate.average_improvement:.4f}")
            perf_lines.append(f"  Success Rate: {best_candidate.success_rate:.2%}")
        
        return "\n".join(perf_lines)
    
    def get_summary(self) -> dict[str, Any]:
        """Get population summary for logging."""
        return {
            "generation": self._generation,
            "population_size": len(self._candidates),
            "candidates": [c.as_dict() for c in self._candidates.values()],
        }

    def save_population(self, path: str) -> None:
        """Save mutation population to JSON."""
        import json
        
        data = {
            "generation": self._generation,
            "candidates": [c.as_dict() for c in self._candidates.values()],
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load_population(cls, path: str, **kwargs) -> "MutationPromptManager":
        """Load mutation population from JSON."""
        import json
        
        data = json.loads(Path(path).read_text())
        candidates = [MutationPromptCandidate.from_dict(c) for c in data["candidates"]]
        manager = cls(candidates=candidates, **kwargs)
        manager._generation = data["generation"]
        return manager