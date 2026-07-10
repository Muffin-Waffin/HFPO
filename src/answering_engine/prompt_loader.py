"""Prompt loading for the Answering Engine.

This module defines :class:`PromptLoader`, responsible for loading prompts
from multiple sources into the unified :class:`Prompt` representation used
by all downstream Answering Engine components. Supported sources include
plain text files, JSON files, JSONL files (including the
``generated_prompts.jsonl`` format produced by the evolution pipeline),
``PromptCandidate`` objects from the evolutionary subsystem, and raw
strings.

The loader is intentionally read-only with respect to the evolutionary
pipeline: it imports ``PromptCandidate`` only to read its ``id`` and
``text`` fields, and never modifies any evolutionary state.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from src.answering_engine.models import Prompt
from src.core.prompt_candidate import PromptCandidate


class PromptLoader:
    """Loads prompts from various sources into :class:`Prompt` objects.

    ``PromptLoader`` is a stateless utility class that provides static
    methods for loading prompts from each supported source type, plus a
    convenience dispatcher (``load``) that auto-detects the source type
    from a file path's extension or an object's type.

    Supported sources:
        - **Text file** (``.txt``): One prompt per file; the filename
          (without extension) is used as the prompt ID.
        - **JSON file** (``.json``): A single prompt dictionary or a
          list of prompt dictionaries. Each dictionary must contain at
          least ``"id"`` and ``"text"`` keys.
        - **JSONL file** (``.jsonl``): One JSON object per line. Supports
          both the ``generated_prompts.jsonl`` format (with a nested
          ``"generated"`` key) and flat ``{"id": ..., "text": ...}``
          format.
        - **PromptCandidate object**: Extracts ``id`` and ``text``
          from an existing evolutionary prompt candidate.
        - **Raw string**: Wraps a bare string with an auto-generated
          UUID as the prompt ID.
    """

    @staticmethod
    def from_text_file(path: str | Path) -> list[Prompt]:
        """Load a single prompt from a plain text file.

        The entire file content is used as the prompt text. The filename
        (without extension) is used as the prompt ID.

        Args:
            path: Path to the text file.

        Returns:
            A list containing a single :class:`Prompt`.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or whitespace-only.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Text file is empty: {path}")

        return [
            Prompt(
                id=path.stem,
                text=text,
                source="text_file",
                metadata={"file": str(path)},
            )
        ]

    @staticmethod
    def from_json_file(path: str | Path) -> list[Prompt]:
        """Load prompts from a JSON file.

        The file may contain either a single prompt dictionary or a
        list of prompt dictionaries. Each dictionary must have at least
        ``"id"`` and ``"text"`` keys.

        Args:
            path: Path to the JSON file.

        Returns:
            A list of :class:`Prompt` objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON structure is not a dict or list.
            KeyError: If a prompt dictionary is missing required keys.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            entries = [data]
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(
                f"JSON file must contain a dict or list, got "
                f"{type(data).__name__}: {path}"
            )

        prompts: list[Prompt] = []
        for entry in entries:
            prompt = PromptLoader._parse_prompt_dict(entry, source="json")
            prompt.metadata["file"] = str(path)
            prompts.append(prompt)

        return prompts

    @staticmethod
    def from_jsonl_file(path: str | Path) -> list[Prompt]:
        """Load prompts from a JSONL file.

        Each line must be a valid JSON object. The method supports
        two formats:

        1. **Generated prompts format** (from the evolution pipeline):
           ``{"generated": {"id": "...", "text": "..."}, ...}``
        2. **Flat format**: ``{"id": "...", "text": "...", ...}``

        Args:
            path: Path to the JSONL file.

        Returns:
            A list of :class:`Prompt` objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If a line cannot be parsed.
        """
        path = Path(path)
        prompts: list[Prompt] = []

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {path}: {e}"
                    ) from e

                # Support generated_prompts.jsonl format.
                if "generated" in data and isinstance(data["generated"], dict):
                    entry = data["generated"]
                    metadata: dict[str, object] = {
                        "file": str(path),
                        "line": line_num,
                    }
                    if "generation" in data:
                        metadata["generation"] = data["generation"]
                    if "origin" in data:
                        metadata["origin"] = data["origin"]

                    prompts.append(
                        Prompt(
                            id=str(entry["id"]),
                            text=str(entry["text"]),
                            source="jsonl",
                            metadata=metadata,
                        )
                    )
                else:
                    prompt = PromptLoader._parse_prompt_dict(
                        data, source="jsonl"
                    )
                    prompt.metadata["file"] = str(path)
                    prompt.metadata["line"] = line_num
                    prompts.append(prompt)

        return prompts

    @staticmethod
    def from_candidate(candidate: PromptCandidate) -> Prompt:
        """Create a Prompt from an existing PromptCandidate.

        Extracts the ``id`` and ``text`` fields from the evolutionary
        candidate without modifying it.

        Args:
            candidate: An evaluated or unevaluated ``PromptCandidate``
                from the evolutionary pipeline.

        Returns:
            A single :class:`Prompt` wrapping the candidate's data.

        Raises:
            TypeError: If ``candidate`` is not a ``PromptCandidate``.
        """
        if not isinstance(candidate, PromptCandidate):
            raise TypeError(
                f"Expected PromptCandidate, got {type(candidate).__name__}."
            )

        return Prompt(
            id=candidate.id,
            text=candidate.text,
            source="candidate",
            metadata={
                "generation": candidate.generation,
                "origin": candidate.origin,
            },
        )

    @staticmethod
    def from_string(text: str, prompt_id: str | None = None) -> Prompt:
        """Create a Prompt from a raw string.

        Args:
            text: The system prompt text.
            prompt_id: Optional identifier. If ``None``, a UUID is
                generated automatically.

        Returns:
            A single :class:`Prompt`.

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
        """
        if not text.strip():
            raise ValueError("Prompt text must not be empty or whitespace.")

        return Prompt(
            id=prompt_id or str(uuid.uuid4()),
            text=text,
            source="string",
        )

    @staticmethod
    def load(source: str | Path | PromptCandidate) -> list[Prompt]:
        """Auto-detect the source type and load prompts.

        Dispatches to the appropriate loading method based on the
        source type:

        - ``PromptCandidate`` → ``from_candidate()``
        - ``str`` or ``Path`` with ``.txt`` extension → ``from_text_file()``
        - ``str`` or ``Path`` with ``.json`` extension → ``from_json_file()``
        - ``str`` or ``Path`` with ``.jsonl`` extension → ``from_jsonl_file()``
        - ``str`` that is not a valid file path → ``from_string()``

        Args:
            source: A file path (str or Path), a ``PromptCandidate``,
                or a raw prompt string.

        Returns:
            A list of :class:`Prompt` objects.

        Raises:
            ValueError: If the source type or file extension is not
                recognized.
        """
        if isinstance(source, PromptCandidate):
            return [PromptLoader.from_candidate(source)]

        path = Path(source) if isinstance(source, str) else source

        if isinstance(source, str) and not path.exists():
            return [PromptLoader.from_string(source)]

        if path.suffix == ".txt":
            return PromptLoader.from_text_file(path)
        elif path.suffix == ".json":
            return PromptLoader.from_json_file(path)
        elif path.suffix == ".jsonl":
            return PromptLoader.from_jsonl_file(path)
        else:
            raise ValueError(
                f"Unrecognized file extension '{path.suffix}' for "
                f"prompt loading. Supported: .txt, .json, .jsonl"
            )

    @staticmethod
    def select_best(prompts: list[Prompt]) -> Prompt:
        """Select the single best prompt from a list of prompts.

        Checks evolutionary artifacts and prompt metadata in the
        following order of precedence:

        1. **Snapshot verification**: Checks if any prompt's ID matches
           the ``best_candidate_id`` recorded in evolutionary snapshots
           (e.g., ``snapshots/generation_*.json``). If multiple match
           different snapshots, the one from the latest snapshot is chosen.
        2. **Metadata fitness**: If no snapshot match exists, checks if any
           prompt has a ``fitness`` or ``fitness_vector`` inside its
           ``metadata``. The prompt with the highest average fitness is chosen.
        3. **Generation number**: Checks ``metadata["generation"]`` and
           picks the prompt with the highest generation number.
        4. **Order fallback**: If all else is equal, picks the last prompt
           in the list (`prompts[-1]`), which is typically the most recently
           generated candidate in sequential logs.

        Args:
            prompts: List of candidate :class:`Prompt` objects.

        Returns:
            The single best :class:`Prompt` object from the list.

        Raises:
            ValueError: If ``prompts`` is empty.
        """
        if not prompts:
            raise ValueError("Cannot select best prompt from an empty list.")
        if len(prompts) == 1:
            return prompts[0]

        # 1. Check evolutionary snapshots across results directories.
        import glob
        best_snapshot_scores: dict[str, float] = {}
        for sf in sorted(glob.glob("**/snapshots/generation_*.json", recursive=True)):
            if "venv" in sf or "brain" in sf or "__pycache__" in sf:
                continue
            try:
                with open(sf, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                bid = sdata.get("best_candidate_id")
                score = sdata.get("best_score")
                if bid and score is not None:
                    best_snapshot_scores[bid] = max(
                        score, best_snapshot_scores.get(bid, -1.0)
                    )
            except Exception:
                pass

        snapshot_matches = [p for p in prompts if p.id in best_snapshot_scores]
        if snapshot_matches:
            # Sort by snapshot best_score descending, then generation descending
            snapshot_matches.sort(
                key=lambda p: (
                    best_snapshot_scores.get(p.id, -1.0),
                    int(p.metadata.get("generation", 0)) if isinstance(p.metadata.get("generation"), (int, float)) else 0,
                ),
                reverse=True,
            )
            return snapshot_matches[0]

        # 2. Check metadata fitness/scores.
        def _get_fitness(p: Prompt) -> float | None:
            if "fitness" in p.metadata and isinstance(p.metadata["fitness"], (int, float)):
                return float(p.metadata["fitness"])
            if "score" in p.metadata and isinstance(p.metadata["score"], (int, float)):
                return float(p.metadata["score"])
            fvec = p.metadata.get("fitness_vector")
            if isinstance(fvec, dict) and fvec:
                vals = [v for v in fvec.values() if isinstance(v, (int, float))]
                if vals:
                    return sum(vals) / len(vals)
            return None

        fitness_pairs = [(p, _get_fitness(p)) for p in prompts if _get_fitness(p) is not None]
        if fitness_pairs:
            fitness_pairs.sort(key=lambda item: item[1], reverse=True)  # type: ignore[arg-type]
            return fitness_pairs[0][0]

        # 3. Check generation number.
        def _get_gen(p: Prompt) -> int:
            gen = p.metadata.get("generation", -1)
            try:
                return int(gen)
            except (ValueError, TypeError):
                return -1

        prompts_sorted = sorted(prompts, key=lambda p: _get_gen(p), reverse=True)
        max_gen = _get_gen(prompts_sorted[0])
        if max_gen >= 0:
            # Return the last prompt among those that share the max generation
            max_gen_prompts = [p for p in prompts if _get_gen(p) == max_gen]
            return max_gen_prompts[-1]

        # 4. Fallback to last prompt in list.
        return prompts[-1]

    @staticmethod
    def filter_prompts(
        prompts: list[Prompt],
        prompt_id: str | None = None,
        prompt_index: int | None = None,
        best_only: bool = False,
    ) -> list[Prompt]:
        """Filter a list of prompts by ID, index, or best-candidate criteria.

        Args:
            prompts: The initial list of loaded prompts.
            prompt_id: If provided, keeps only prompts whose ID equals
                or starts with this string (case-insensitive).
            prompt_index: If provided, keeps only the prompt at this
                index (supports negative indexing like ``-1`` for last).
            best_only: If ``True``, selects the single best prompt using
                :meth:`select_best`.

        Returns:
            A filtered list of :class:`Prompt` objects.

        Raises:
            ValueError: If filtering results in an empty list or out-of-bounds index.
        """
        if not prompts:
            return []

        filtered = prompts

        if prompt_index is not None:
            if not (-len(filtered) <= prompt_index < len(filtered)):
                raise ValueError(
                    f"Prompt index {prompt_index} is out of bounds for "
                    f"list of {len(filtered)} prompts."
                )
            filtered = [filtered[prompt_index]]

        if prompt_id is not None:
            pid_clean = prompt_id.strip().lower()
            matches = [
                p for p in filtered
                if p.id.lower() == pid_clean or p.id.lower().startswith(pid_clean)
            ]
            if not matches:
                raise ValueError(f"No prompt found matching ID '{prompt_id}'.")
            filtered = matches

        if best_only and len(filtered) > 1:
            best = PromptLoader.select_best(filtered)
            print(
                f"[PromptLoader] Selected best prompt: ID={best.id[:12]}... "
                f"(Gen={best.metadata.get('generation', 'N/A')})"
            )
            filtered = [best]

        return filtered

    @staticmethod
    def _parse_prompt_dict(
        data: dict[str, Any],
        source: str,
    ) -> Prompt:
        """Parse a dictionary into a Prompt.

        Supports dictionaries with ``"id"`` and ``"text"`` keys, as
        well as full ``PromptCandidate.as_dict()`` format.

        Args:
            data: A dictionary with at least ``"id"`` and ``"text"`` keys.
            source: The source label to assign (e.g., ``"json"``).

        Returns:
            A :class:`Prompt` instance.

        Raises:
            KeyError: If ``"id"`` or ``"text"`` keys are missing.
        """
        return Prompt(
            id=str(data["id"]),
            text=str(data["text"]),
            source=source,
            metadata={
                k: v for k, v in data.items()
                if k not in ("id", "text")
            },
        )
