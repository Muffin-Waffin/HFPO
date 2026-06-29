from datasets import load_dataset

LETTER_TO_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


def _preprocess(example):
    return {
        "question": example["question"],
        "context": "",
        "choices": [
            example["options"]["A"],
            example["options"]["B"],
            example["options"]["C"],
            example["options"]["D"],
        ],
        "answer": LETTER_TO_INDEX[example["answer_idx"]],
    }


def load_medqa(split: str = "train"):

    dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split=split,
    )

    dataset = dataset.map(
        _preprocess,
        remove_columns=dataset.column_names,
    )

    return dataset