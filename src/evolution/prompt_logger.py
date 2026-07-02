"""
Prompt generation logger for HFPO.

Records every generated prompt together with its parents into a JSONL file.
Each line is an independent JSON object so logging is append-only and safe
for long-running evolutionary experiments.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.core.prompt_candidate import PromptCandidate


class PromptLogger:
    """Append generated prompt information to a JSONL log."""

    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        generation: int,
        child: PromptCandidate,
        parent_a: PromptCandidate,
        parent_b: PromptCandidate | None,
        
    ) -> None:
        """Append one generated prompt to the log."""

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "generation": generation,
            "candidate_id": child.id,
            "origin": child.origin,
            "parent_a": {
                "id": parent_a.id,
                "text": parent_a.text,
            },
            "parent_b": (
                {
                    "id": parent_b.id,
                    "text": parent_b.text,
                }
                if parent_b is not None
                else None
            ),
            "metadata": child.metadata,
            "generated": {
                "id": child.id,
                "text": child.text,
            },
        }

        with self._path.open("a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=None)
            file.write("\n")