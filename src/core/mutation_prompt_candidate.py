"""Lightweight mutation strategy candidate for adaptive mutation population."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass(slots=True)
class MutationPromptCandidate:
    """Represents a single mutation strategy in the adaptive population.

    Unlike PromptCandidate, this is NOT a task prompt. It is a compact
    mutation strategy that gets injected into a fixed mutation template.
    """

    id: str
    strategy: str                 # The compact mutation strategy text
    generation: int
    parent_ids: list[str]
    age: int = 0                  # Generations since creation
    times_selected: int = 0       # How many times this strategy was chosen

    # Child outcome statistics
    children_generated: int = 0
    children_improved: int = 0
    children_worsened: int = 0
    sum_improvement: float = 0.0
    improvements: list[float] = field(default_factory=list)  # All individual improvements

    # Top children tracking
    top_successful_children: list[dict[str, Any]] = field(default_factory=list)  # Best improvements
    top_failed_children: list[dict[str, Any]] = field(default_factory=list)      # Worst degradations

    # Computed fitness (set by ranking strategy)
    fitness: float | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.strategy.strip():
            raise ValueError("Mutation strategy text must not be empty.")
        if self.generation < 0:
            raise ValueError(f"Generation must be >= 0, got {self.generation}.")

    @property
    def success_rate(self) -> float:
        if self.children_generated == 0:
            return 0.0
        return self.children_improved / self.children_generated

    @property
    def average_improvement(self) -> float:
        if self.children_generated == 0:
            return 0.0
        return self.sum_improvement / self.children_generated

    @property
    def median_improvement(self) -> float:
        if not self.improvements:
            return 0.0
        sorted_imps = sorted(self.improvements)
        n = len(sorted_imps)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_imps[mid - 1] + sorted_imps[mid]) / 2.0
        return sorted_imps[mid]

    @property
    def best_improvement(self) -> float:
        if not self.improvements:
            return 0.0
        return max(self.improvements)

    @property
    def worst_improvement(self) -> float:
        if not self.improvements:
            return 0.0
        return min(self.improvements)

    def record_child_result(
        self,
        parent_id: str,
        child_id: str,
        parent_fitness: float,
        child_fitness: float,
        generation: int,
    ) -> None:
        """Update statistics from one generated child."""
        improvement = child_fitness - parent_fitness
        self.children_generated += 1
        self.sum_improvement += improvement
        self.improvements.append(improvement)
        self.times_selected += 1

        if improvement > 0:
            self.children_improved += 1
            self.top_successful_children.append({
                "parent_id": parent_id,
                "child_id": child_id,
                "parent_fitness": parent_fitness,
                "child_fitness": child_fitness,
                "improvement": improvement,
                "generation": generation,
            })
        elif improvement < 0:
            self.children_worsened += 1
            self.top_failed_children.append({
                "parent_id": parent_id,
                "child_id": child_id,
                "parent_fitness": parent_fitness,
                "child_fitness": child_fitness,
                "improvement": improvement,
                "generation": generation,
            })

        # Keep only top 5 in each category
        self.top_successful_children.sort(key=lambda x: x["improvement"], reverse=True)
        self.top_failed_children.sort(key=lambda x: x["improvement"])
        self.top_successful_children = self.top_successful_children[:5]
        self.top_failed_children = self.top_failed_children[:5]

    def increment_age(self) -> None:
        self.age += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "age": self.age,
            "times_selected": self.times_selected,
            "children_generated": self.children_generated,
            "children_improved": self.children_improved,
            "children_worsened": self.children_worsened,
            "sum_improvement": self.sum_improvement,
            "average_improvement": self.average_improvement,
            "median_improvement": self.median_improvement,
            "best_improvement": self.best_improvement,
            "worst_improvement": self.worst_improvement,
            "success_rate": self.success_rate,
            "fitness": self.fitness,
            "top_successful_children": self.top_successful_children,
            "top_failed_children": self.top_failed_children,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MutationPromptCandidate":
        candidate = cls(
            id=data["id"],
            strategy=data["strategy"],
            generation=data["generation"],
            parent_ids=data["parent_ids"],
            age=data.get("age", 0),
            times_selected=data.get("times_selected", 0),
            children_generated=data.get("children_generated", 0),
            children_improved=data.get("children_improved", 0),
            children_worsened=data.get("children_worsened", 0),
            sum_improvement=data.get("sum_improvement", 0.0),
            improvements=data.get("improvements", []),
            top_successful_children=data.get("top_successful_children", []),
            top_failed_children=data.get("top_failed_children", []),
            fitness=data.get("fitness"),
            metadata=data.get("metadata", {}),
        )
        return candidate

    def __str__(self) -> str:
        return (
            f"MutationPromptCandidate(id={self.id[:8]}, gen={self.generation}, "
            f"fitness={self.fitness:.4f}, avg_imp={self.average_improvement:.4f}, "
            f"success_rate={self.success_rate:.2f}, selected={self.times_selected})"
        )