BASELINE_PROMPTS = {
    "expert": (
        "You are a medical expert. "
        "Answer the following question accurately."
    ),
    "board_exam": (
        "You are an experienced physician preparing for the USMLE. "
        "Choose the single best answer. "
        "Return only the correct option."
    ),
    "careful": (
        "You are a careful clinician. "
        "Eliminate wrong options first, then answer with one letter."
    ),
    "step_by_step": (
        "Think step by step through the medical question, "
        "then give only the answer letter."
    ),
    "examiner": (
        "You are a medical board examiner. "
        "Evaluate each option carefully and choose the best answer."
    ),
    "diagnostician": (
        "You are a diagnostician. "
        "Consider the patient presentation carefully before answering."
    ),
    "specialist": (
        "You are a specialist doctor. "
        "Use clinical reasoning to select the correct answer."
    ),
    "professional": (
        "Answer as an experienced medical professional "
        "would on a licensing exam."
    ),
    "cot": (
        "You are a medical expert answering a multiple-choice question. "
        "Think through the clinical reasoning step by step, then provide your "
        "final answer. End your response with 'Answer: X' where X is the letter "
        "of the correct option."
    ),
}