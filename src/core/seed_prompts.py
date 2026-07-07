"""Seed prompts for the initial HFPO population.

This module defines the initial population of system prompts used to
bootstrap the genetic algorithm. Each prompt represents a different
reasoning style for answering multiple-choice medical questions. These
prompts are converted into generation-0 PromptCandidate objects by
the experiment runner before evolutionary optimization begins.
"""

SEED_PROMPTS = [
    "Answer the question.",
    "Think before answering.",
    "Use your medical knowledge.",
    "Read carefully.",
    "Choose one option.",
    "Explain your reasoning internally before answering.",
    "Focus on the clinical findings.",
    "Eliminate incorrect options first.",
    "Treat this as a medical licensing exam.",
    "Act as an experienced physician.",
    "Prioritize patient safety.",
    "Base your answer only on the information provided.",
    "Consider alternative diagnoses.",
    "Be concise.",
    "Be systematic.",
    "Use evidence rather than intuition.",
    "Identify the key clue.",
    "Double-check your conclusion.",
    "Think step by step.",
    "Make the best clinical decision."
]

# SEED_PROMPTS: list[str] = [
#     (
#         "You are a doctor. Answer the question."
#     ),
#     (
#         "Choose the correct answer for the following medical question."
#     ),
#     (
#         "You are taking a medical licensing examination. Select the best answer."
#     ),
#     (
#         "You are evaluating a patient. Choose the most appropriate diagnosis or management option."
#     ),
#     (
#         "Use your medical knowledge to answer the question."
#     ),
#     (
#         "Answer accurately."
#     ),
#     (
#         "Choose the safest and most appropriate medical answer."
#     ),
#     (
#         "Make the best clinical decision based on the information provided."
#     ),
#     (
#         "Identify the most likely diagnosis or next best step."
#     ),
#     (
#         "Think briefly before selecting the best answer."
#     ),
    # (
    #     "Think like a physician solving an unfamiliar case."
    # ),
    # (
    #     "Be cautious and avoid making unsupported assumptions."
    # ),
    # (
    #     "Identify the key medical clue before choosing an answer."
    # ),
    # (
    #     "Consider each option before making a final decision."
    # ),
    # (
    #     "Prioritize patient safety when selecting the answer."
    # ),
    # (
    #     "Choose the answer that is best supported by the information given."
    # ),
    # (
    #     "Solve the question systematically rather than relying on intuition."
    # ),
    # (
    #     "Focus only on the evidence presented in the question."
    # ),
    # (
    #     "Decide which option is most medically justified."
    # ),
    # (
    #     "Answer as accurately as possible using sound clinical judgment."
    # ),
# ]