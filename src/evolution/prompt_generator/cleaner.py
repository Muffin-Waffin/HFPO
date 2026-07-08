"""Text-cleaning utility for the Prompt Generator subsystem.

This module defines PromptCleaner, which is responsible only for
transforming raw text returned by a reasoning LLM into normalized
prompt text. It performs no validation, no duplicate detection, no
PromptCandidate creation, no LLM interaction, and no template
generation.
"""

import re


class PromptCleaner:
    """Cleans raw LLM output into normalized prompt text.

    PromptCleaner applies a fixed, ordered pipeline of textual
    transformations to raw text returned by a reasoning LLM: newline
    normalization, code-fence removal, Markdown heading-line removal,
    numbering removal, bullet-prefix removal, blank-line collapsing,
    surrounding-quote removal, and whitespace trimming. Quote removal
    runs after formatting artifacts (headings, numbering, bullets,
    repeated blank lines) have already been stripped, since by that
    point any surrounding quotes are far more likely to wrap the
    entire remaining prompt rather than a sub-fragment of it.

    PromptCleaner performs no validation, duplicate detection,
    candidate construction, LLM interaction, or template generation;
    those responsibilities belong to other components such as
    PromptValidator. It is conservative by design: it removes
    formatting artifacts, not semantic content, and never deletes a
    line solely for resembling a label (e.g. "Instructions" or
    "Prompt:") unless that line is an explicit Markdown heading
    beginning with ``#``.
    """

    __slots__ = ()
    def __call__(self, text: str) -> str:
        """Clean raw prompt text.

        Args:
            text: Raw text returned by a reasoning LLM.

        Returns:
            The cleaned prompt text.
        """
        return self.clean(text)

    def clean(self, text: str) -> str:
        """Cleans raw LLM output text into normalized prompt text.

        Runs the full cleaning pipeline in order: newline
        normalization, code-fence removal, Markdown heading-line
        removal, numbering removal, bullet-prefix removal, blank-line
        collapsing, surrounding-quote removal, and whitespace
        trimming.

        Args:
            text: Raw text returned by a reasoning LLM.

        Returns:
            The cleaned prompt text.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"text must be a string, got {type(text).__name__}."
            )

        result = text
        result = self._normalize_newlines(result)
        result = self._remove_code_fences(result)
        result = self._remove_markdown_headings(result)
        result = self._remove_numbering(result)
        result = self._remove_bullets(result)
        result = self._collapse_blank_lines(result)
        result = self._remove_surrounding_quotes(result)
        result = self._trim_lines(result)
        return result.strip()

    def _normalize_newlines(self, text: str) -> str:
        """Normalizes all line endings to LF.

        Converts CRLF (``\\r\\n``) and CR (``\\r``) line endings to LF
        (``\\n``).

        Args:
            text: Text whose line endings should be normalized.

        Returns:
            The text with all line endings converted to LF.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_code_fences(self, text: str) -> str:
        """Removes Markdown code fence delimiters.

        Strips triple-backtick fences, including variants with arbitrary
        language tags (for example ```` ```python ````, ```` ```text ````,
        or ```` ```jsonc ````), leaving the fenced content in place.

        Args:
            text: Text that may contain Markdown code fences.

        Returns:
            The text with code fence delimiters removed.
        """
        return re.sub(r"```[^\n]*\n?|```", "", text)

    def _remove_markdown_headings(self, text: str) -> str:
        """Removes Markdown heading lines entirely.

        Removes an entire line, including its heading text, whenever
        that line begins with one to six ``#`` characters followed
        by whitespace and heading text (for example ``# Prompt`` or
        ``## Instructions``). Headings are formatting artifacts and
        are not part of the prompt content, so the whole line is
        discarded rather than just the ``#`` markers.

        Args:
            text: Text that may contain Markdown headings.

        Returns:
            The text with heading lines removed.
        """
        return re.sub(r"(?m)^#{1,6}[ \t]+.*$\n?", "", text)

    def _remove_numbering(self, text: str) -> str:
        """Removes leading numbering prefixes at the start of lines.

        Strips numbering formats such as ``1.``, ``2.``, ``10.``,
        ``(1)``, or ``1)`` -- along with any trailing spaces or tabs
        and the line's terminating newline -- only when they occur at
        the beginning of a line.

        Args:
            text: Text that may contain numbered-list prefixes.

        Returns:
            The text with leading numbering prefixes removed.
        """
        return re.sub(r"(?m)^(\(\d+\)|\d+[.)])[ \t]*\n?", "", text)

    def _remove_bullets(self, text: str) -> str:
        """Removes bullet-list prefixes at the start of lines.

        Strips a leading ``-``, ``*``, or ``\u2022`` (and any
        following spaces or tabs) only when it occurs at the beginning
        of a line.

        Args:
            text: Text that may contain bullet-prefixed lines.

        Returns:
            The text with leading bullet prefixes removed.
        """
        return re.sub(r"(?m)^[\-\*\u2022][ \t]*", "", text)

    def _collapse_blank_lines(self, text: str) -> str:
        """Collapses consecutive blank lines into a single blank line.

        Replaces any run of two or more consecutive blank lines,
        including lines containing only spaces or tabs, with exactly
        one blank line.

        Args:
            text: Text that may contain repeated blank lines.

        Returns:
            The text with repeated blank lines collapsed to at most
            one.
        """
        return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

    def _remove_surrounding_quotes(self, text: str) -> str:
        """Removes quotes surrounding the entire text.

        Strips a single matching pair of surrounding quote characters
        -- ``"``, ``'``, triple single quotes (``'''``), or triple
        double quotes (``\"\"\"``) -- only when they wrap the whole
        (stripped) text. Quotes appearing inside the text are never
        removed.

        Args:
            text: Text that may be wrapped in surrounding quotes.

        Returns:
            The text with one layer of surrounding quotes removed, if
            present.
        """
        stripped = text.strip()

        for quote in ('"""', "'''", '"', "'"):
            length = len(quote)
            if (
                len(stripped) >= 2 * length
                and stripped.startswith(quote)
                and stripped.endswith(quote)
            ):
                return stripped[length:-length].strip()
            
        PREFIXES = [
            "Here is the new prompt:",
            "Here is the prompt:",
            "New prompt:",
            "Prompt:",
            "Offspring prompt:",
        ]
   
        for prefix in PREFIXES:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].lstrip()

        return text

    def _trim_lines(self, text: str) -> str:
        """Removes trailing spaces from each line.

        Strips trailing whitespace from the end of every line while
        preserving the line structure of the text.

        Args:
            text: Text whose lines may contain trailing spaces.

        Returns:
            The text with trailing spaces removed from every line.
        """
        return "\n".join(line.rstrip() for line in text.split("\n"))