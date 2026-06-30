"""Persistence utility for HFPO experiment outputs.

This module defines :class:`OutputManager`, a purely I/O-oriented
component responsible for writing generation snapshots, experiment
summaries, best-prompt reports, and population checkpoints to disk.

OutputManager performs no mutation, crossover, prompt generation,
evaluation, tournament selection, population management, or federated
communication of any kind. It is exclusively a file-persistence
utility used by the rest of the HFPO evolutionary pipeline (e.g.
ServerEvolutionEngine, GenerationLogger) to durably record state.

Typical usage:

    >>> manager = OutputManager(output_directory="outputs")
    >>> manager.save_snapshot(snapshot)
    >>> manager.save_best_prompt(best_candidate)
    >>> manager.save_summary(all_snapshots)
    >>> manager.save_checkpoint(population, generation=5)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.population import Population
from src.core.prompt_candidate import PromptCandidate
from src.evolution.generation_snapshot import GenerationSnapshot


class OutputManager:
    """Persists HFPO experiment artifacts to disk.

    OutputManager is a narrow, single-responsibility utility: it knows
    how to write generation snapshots, best-prompt reports, experiment
    summaries, and population checkpoints in a consistent, predictable
    on-disk layout. It holds no evolutionary logic and no federated
    state; it only serializes data it is handed and writes it to the
    appropriate file.

    Attributes:
        _output_directory: Root directory under which all experiment
            outputs are written.
        _snapshot_directory: Subdirectory holding per-generation
            snapshot JSON files.
        _best_prompt_directory: Subdirectory holding the best-prompt
            text report.
        _checkpoint_directory: Subdirectory holding population
            checkpoint JSON files.
    """

    __slots__ = (
        "_output_directory",
        "_snapshot_directory",
        "_best_prompt_directory",
        "_checkpoint_directory",
    )

    def __init__(self, output_directory: str = "outputs") -> None:
        """Initializes the OutputManager and creates its directory tree.

        Creates the root output directory (if it does not already
        exist) along with the three subdirectories used by this class:
        ``snapshots/``, ``best_prompts/``, and ``checkpoints/``.

        Args:
            output_directory: Path to the root directory under which
                all experiment outputs will be written. Defaults to
                ``"outputs"``.

        Raises:
            TypeError: If ``output_directory`` is not a string.
        """
        if not isinstance(output_directory, str):
            raise TypeError("output_directory must be a string.")

        self._output_directory = Path(output_directory)
        self._snapshot_directory = self._output_directory / "snapshots"
        self._best_prompt_directory = self._output_directory / "best_prompts"
        self._checkpoint_directory = self._output_directory / "checkpoints"

        self._output_directory.mkdir(parents=True, exist_ok=True)
        self._snapshot_directory.mkdir(parents=True, exist_ok=True)
        self._best_prompt_directory.mkdir(parents=True, exist_ok=True)
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_snapshot(self, snapshot: GenerationSnapshot) -> Path:
        """Writes a single generation snapshot to disk.

        The snapshot is serialized via its ``to_dict()`` method and
        written to ``snapshots/generation_{N:03d}.json``, where ``N``
        is the snapshot's generation index. Any existing file at that
        path is overwritten.

        Args:
            snapshot: The GenerationSnapshot to persist.

        Returns:
            The Path of the file that was written.

        Raises:
            TypeError: If ``snapshot`` is not a GenerationSnapshot
                instance.
            ValueError: If the snapshot's generation is negative.
        """
        if not isinstance(snapshot, GenerationSnapshot):
            raise TypeError("snapshot must be a GenerationSnapshot instance.")
        if snapshot.generation < 0:
            raise ValueError("snapshot.generation must be non-negative.")

        path = self._snapshot_directory / f"generation_{snapshot.generation:03d}.json"
        self._write_json(path, snapshot.to_dict())
        return path

    def save_best_prompt(self, candidate: PromptCandidate) -> Path:
        """Writes a human-readable report for the best prompt found.

        Writes ``best_prompts/best_prompt.txt`` containing the
        candidate's ID, generation, origin, and prompt text. Any
        existing file at that path is overwritten.

        Args:
            candidate: The PromptCandidate to report on.

        Returns:
            The Path of the file that was written.

        Raises:
            TypeError: If ``candidate`` is not a PromptCandidate
                instance.
            ValueError: If ``candidate.generation`` is negative.
        """
        if not isinstance(candidate, PromptCandidate):
            raise TypeError("candidate must be a PromptCandidate instance.")
        if candidate.generation < 0:
            raise ValueError("candidate.generation must be non-negative.")

        report_lines = [
            f"Candidate ID: {candidate.id}",
            f"Generation: {candidate.generation}",
            f"Origin: {candidate.origin}",
            "",
            "Prompt:",
            str(candidate.text),
        ]
        text = "\n".join(report_lines)

        path = self._best_prompt_directory / "best_prompt.txt"
        self._write_text(path, text)
        return path

    def save_summary(self, snapshots: list[GenerationSnapshot]) -> Path:
        """Writes an aggregate experiment summary to disk.

        Writes ``summary.json`` containing the total number of
        generations recorded, the generation index that achieved the
        best score, that best score itself, the final generation
        index, and the score of the final (highest-generation)
        snapshot.

        Args:
            snapshots: A non-empty list of GenerationSnapshot
                instances.

        Returns:
            The Path of the file that was written.

        Raises:
            TypeError: If ``snapshots`` is not a list, or if any
                element is not a GenerationSnapshot instance.
            ValueError: If ``snapshots`` is empty.
        """
        if not isinstance(snapshots, list):
            raise TypeError("snapshots must be a list of GenerationSnapshot instances.")
        if len(snapshots) == 0:
            raise ValueError("snapshots must not be empty.")
        for snapshot in snapshots:
            if not isinstance(snapshot, GenerationSnapshot):
                raise TypeError("snapshots must contain only GenerationSnapshot instances.")

        ordered_snapshots = sorted(snapshots, key=lambda s: s.generation)
        best_snapshot = max(snapshots, key=lambda s: s.best_score)
        final_snapshot = ordered_snapshots[-1]

        summary_data = {
            "total_generations": len(ordered_snapshots),
            "best_generation": best_snapshot.generation,
            "best_score": best_snapshot.best_score,
            "final_generation": final_snapshot.generation,
            "final_score": final_snapshot.best_score,
        }

        path = self._output_directory / "summary.json"
        self._write_json(path, summary_data)
        return path

    def save_checkpoint(self, population: Population, generation: int) -> Path:
        """Writes a lightweight population checkpoint to disk.

        Writes ``checkpoints/checkpoint_{N:03d}.json``, where ``N`` is
        the supplied generation. Only candidate IDs (and the resulting
        population size) are serialized; full PromptCandidate objects
        and their FitnessVectors are never written.

        Args:
            population: The Population to checkpoint.
            generation: The generation index this checkpoint
                corresponds to. Must be a non-negative integer.

        Returns:
            The Path of the file that was written.

        Raises:
            TypeError: If ``population`` is not a Population instance,
                or ``generation`` is not an integer.
            ValueError: If ``generation`` is negative.
        """
        if not isinstance(population, Population):
            raise TypeError("population must be a Population instance.")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise TypeError("generation must be an integer.")
        if generation < 0:
            raise ValueError("generation must be non-negative.")

        candidate_ids: list[str] = [candidate.id for candidate in population.candidates]

        checkpoint_data = {
            "generation": generation,
            "population_size": len(candidate_ids),
            "candidate_ids": candidate_ids,
        }

        path = self._checkpoint_directory / f"checkpoint_{generation:03d}.json"
        self._write_json(path, checkpoint_data)
        return path

    def summary(self) -> dict:
        """Returns the directory layout managed by this instance.

        Returns:
            A dict with keys ``output_directory``,
            ``snapshot_directory``, ``checkpoint_directory``, and
            ``best_prompt_directory``, each mapped to its string path.
        """
        return {
            "output_directory": str(self._output_directory),
            "snapshot_directory": str(self._snapshot_directory),
            "checkpoint_directory": str(self._checkpoint_directory),
            "best_prompt_directory": str(self._best_prompt_directory),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_json(
            self,
            path: Path,
            data: dict[str, object],
        ) -> None:
        """Writes a dict to disk as pretty-printed JSON.

        Always overwrites any existing file at ``path``; no append
        mode is used.

        Args:
            path: Destination file path.
            data: JSON-serializable dict to write.
        """
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(data, file_handle, indent=4, sort_keys=True,ensure_ascii=False)

    def _write_text(self, path: Path, text: str) -> None:
        """Writes a plain-text string to disk.

        Always overwrites any existing file at ``path``; no append
        mode is used.

        Args:
            path: Destination file path.
            text: Text content to write.
        """
        with path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(text)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Returns a human-readable string representation.

        Returns:
            A string of the form
            ``OutputManager(output_directory="outputs")``.
        """
        return f'OutputManager(output_directory="{self._output_directory}")'