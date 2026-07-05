"""
Mutation prompt generation logger for HFPO.

Records every generated child with its mutation prompt and parent info into a JSONL file.
Each line is an independent JSON object so logging is append-only and safe
for long-running evolutionary experiments.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.core.prompt_candidate import PromptCandidate
from src.core.mutation_prompt_candidate import MutationPromptCandidate


class MutationPromptLogger:
    """Append mutation prompt generation information to a JSONL log."""

    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        generation: int,
        mutation_prompt: MutationPromptCandidate,
        parent: PromptCandidate,
        child: PromptCandidate,
        parent_fitness: float,
        child_fitness: float,
        improvement: float,
        success: bool,
    ) -> None:
        """Append one mutation generation record to the log."""

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "generation": generation,
            "mutation_prompt_id": mutation_prompt.id,
            "mutation_strategy": mutation_prompt.strategy,
            "parent_prompt_id": parent.id,
            "child_prompt_id": child.id,
            "parent_fitness": parent_fitness,
            "child_fitness": child_fitness,
            "improvement": improvement,
            "success": success,
        }

        with self._path.open("a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=None)
            file.write("\n")