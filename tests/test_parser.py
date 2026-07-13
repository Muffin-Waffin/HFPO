"""
Unit tests for answer parser (src/evaluation/parser.py).

Tests both CoT-style responses (with "Answer:" anchor) and
legacy non-CoT responses (direct letter/word answer).
"""

import pytest
from src.evaluation.parser import parse_multiple_choice, parse_yes_no_maybe


class TestParseMultipleChoice:
    """Tests for parse_multiple_choice function."""

    def test_cot_style_answer_anchor_last(self):
        """CoT response with reasoning then final Answer: X anchor - should pick LAST."""
        response = (
            "A 45-year-old man presents with chest pain. "
            "The differential includes MI, PE, and aortic dissection. "
            "Based on the ECG findings, the most likely diagnosis is MI. "
            "Answer: A"
        )
        assert parse_multiple_choice(response) == 0  # A -> 0

    def test_cot_style_multiple_answer_anchors_picks_last(self):
        """Multiple Answer: patterns in CoT - should pick the final one."""
        response = (
            "Let me think... Option A seems correct initially. "
            "Answer: A\n\nBut wait, re-reading the question, option B is better. "
            "Answer: B"
        )
        assert parse_multiple_choice(response) == 1  # B -> 1 (last Answer:)

    def test_cot_style_with_parentheses(self):
        """Answer: (X) format should work."""
        response = "After careful analysis, the answer is (C). Answer: (C)"
        assert parse_multiple_choice(response) == 2  # C -> 2

    def test_cot_case_insensitive(self):
        """Answer: anchor should be case-insensitive."""
        response = "reasoning here... answer: d"
        assert parse_multiple_choice(response) == 3  # D -> 3

    def test_non_cot_direct_letter(self):
        """Legacy non-CoT: just the letter (backward compatibility)."""
        response = "A"
        assert parse_multiple_choice(response) == 0

    def test_non_cot_letter_in_sentence(self):
        """Legacy: letter embedded in short sentence."""
        response = "The answer is B."
        assert parse_multiple_choice(response) == 1

    def test_non_cot_first_letter_wins(self):
        """Legacy: first standalone letter wins if no Answer: anchor."""
        response = "A is wrong. B is correct."
        assert parse_multiple_choice(response) == 0  # First 'A' (backward compat)

    def test_cot_answer_anchor_beats_first_letter(self):
        """Explicit Answer: anchor should take precedence over early letter."""
        response = "A 45-year-old man presents... Answer: C"
        assert parse_multiple_choice(response) == 2  # C from Answer:, not first A

    def test_empty_response(self):
        """Empty response returns None."""
        assert parse_multiple_choice("") is None

    def test_no_valid_answer(self):
        """Response without any valid letter returns None."""
        response = "I don't know the answer to this question."
        assert parse_multiple_choice(response) is None

    def test_answer_anchor_with_extra_text(self):
        """Answer: X with trailing punctuation/whitespace."""
        response = "Step by step reasoning... Answer: B  "
        assert parse_multiple_choice(response) == 1

    def test_lowercase_letter_fallback(self):
        """Fallback should handle lowercase letters too."""
        response = "the answer is c"
        assert parse_multiple_choice(response) == 2


class TestParseYesNoMaybe:
    """Tests for parse_yes_no_maybe function."""

    def test_cot_style_answer_anchor_last(self):
        """CoT with Answer: yes/no/maybe anchor - picks last."""
        response = (
            "The patient has symptoms consistent with... "
            "Answer: yes"
        )
        assert parse_yes_no_maybe(response) == "yes"

    def test_cot_multiple_anchors_picks_last(self):
        """Multiple Answer: anchors in CoT - picks final."""
        response = (
            "Initially I thought maybe. "
            "Answer: maybe\n\nBut reconsidering... "
            "Answer: no"
        )
        assert parse_yes_no_maybe(response) == "no"

    def test_cot_case_insensitive(self):
        """Answer: anchor case-insensitive."""
        response = "Reasoning... ANSWER: MAYBE"
        assert parse_yes_no_maybe(response) == "maybe"

    def test_non_cot_direct_yes(self):
        """Legacy: direct 'yes' substring."""
        response = "yes"
        assert parse_yes_no_maybe(response) == "yes"

    def test_non_cot_in_sentence(self):
        """Legacy: yes/no/maybe in sentence."""
        response = "The correct answer is no."
        assert parse_yes_no_maybe(response) == "no"

    def test_non_cot_first_match_wins(self):
        """Legacy: first substring match wins (yes before no before maybe)."""
        response = "yes but actually no"
        assert parse_yes_no_maybe(response) == "yes"  # 'yes' appears first

    def test_empty_response(self):
        assert parse_yes_no_maybe("") is None

    def test_no_valid_answer(self):
        response = "I am uncertain about this case."
        assert parse_yes_no_maybe(response) is None

    def test_answer_anchor_beats_substring(self):
        """Explicit Answer: should win over earlier substring."""
        response = "The patient says yes to treatment. Answer: no"
        assert parse_yes_no_maybe(response) == "no"

    def test_answer_anchor_with_extra_text(self):
        response = "Final answer: Answer: maybe   "
        assert parse_yes_no_maybe(response) == "maybe"

    def test_answer_anchor_parentheses(self):
        """Answer: (yes) format - not explicitly supported but let's see"""
        response = "Answer: (yes)"
        # Current regex expects word boundary after answer, so this returns None
        # and falls back to substring -> "yes"
        result = parse_yes_no_maybe(response)
        assert result in ("yes", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])