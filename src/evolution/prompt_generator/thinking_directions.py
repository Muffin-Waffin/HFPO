"""Reasoning methodologies used as mutation operators.

Each mutation samples one reasoning methodology at random and asks
the reasoning LLM to rewrite the parent prompt so that it naturally
encourages that reasoning behavior while preserving the original
medical multiple-choice question-answering task.
"""

__all__ = ["MUTATION_METHODS"]

MUTATION_METHODS: dict[str, str] = {
    "Chain of Thought":
        "Break down the medical reasoning into clear, logical steps. "
        "Encourage transparent reasoning before selecting the final answer.",

    "Trigger Chain of Thought":
        "Guide reasoning through an explicit sequence: identify key clinical findings, "
        "apply relevant medical knowledge, evaluate the answer choices, and then choose the best answer.",

    "Self Consistency":
        "Encourage considering multiple independent reasoning paths and "
        "select the answer that remains most consistent across them.",

    "Tree of Thoughts":
        "Encourage exploring multiple plausible diagnostic or management paths "
        "before converging on the strongest conclusion.",

    "Metacognitive Prompting":
        "Encourage reflecting on assumptions, reasoning strategy, and confidence "
        "before committing to the final answer.",

    "Uncertainty-Based Prompting":
        "Encourage recognizing ambiguity, avoiding unsupported assumptions, and "
        "preferring conclusions supported by the available evidence.",

    "Role-Based Prompting":
        "Encourage reasoning from the perspective of an experienced medical "
        "specialist most appropriate for the clinical scenario.",

    "Guided Prompting":
        "Encourage structured reasoning by focusing attention on the important "
        "clinical findings, differential diagnoses, and evidence supporting each option.",
}



















































# THINKING_DIRECTIONS = [

#     "Rewrite the prompt so the model identifies the key clinical findings before answering.",

#     "Rewrite the prompt so the model first generates a differential diagnosis before selecting the final answer.",

#     "Rewrite the prompt so the model systematically eliminates incorrect options before choosing one.",

#     "Rewrite the prompt so the model prioritizes patient safety when multiple answers appear plausible.",

#     "Rewrite the prompt from the perspective of an experienced attending physician supervising a trainee.",

#     "Rewrite the prompt from the perspective of a clinician working under time pressure in the emergency department.",

#     "Rewrite the prompt so the model verifies its conclusion before producing the final answer.",

#     "Rewrite the prompt so the model reasons from symptoms to diagnosis rather than from diagnosis to symptoms.",

#     "Rewrite the prompt so the model focuses on distinguishing similar diseases with overlapping presentations.",

#     "Rewrite the prompt so the model gives greater weight to the most clinically significant findings.",

#     "Rewrite the prompt using a completely different instructional style while preserving the original objective.",

#     "Rewrite the prompt so the model relies on established medical guidelines and standard clinical practice.",

#     "Rewrite the prompt so the instruction is shorter while preserving all essential behavior.",

#     "Rewrite the prompt so it is more robust to ambiguous or incomplete clinical information.",

#     "Rewrite the prompt so it encourages careful consideration without unnecessary verbosity.",

#     "Rewrite the prompt so the model evaluates every answer option before making a decision.",

#     "Rewrite the prompt so the model actively questions its initial intuition before selecting an answer.",

#     "Rewrite the prompt so the model distinguishes common conditions from dangerous alternatives.",

#     "Rewrite the prompt so the model minimizes overconfidence when evidence is limited.",

#     "Rewrite the prompt using a novel reasoning strategy that is substantially different from the original prompt while preserving the same task."

# ]