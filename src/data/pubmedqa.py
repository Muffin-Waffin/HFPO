from datasets import load_dataset


def _preprocess(example):
    return {
        "question": example["question"],
        "context": "\n".join(example["context"]["contexts"]),
        "choices": [
            "yes",
            "no",
            "maybe",
        ],
        "answer": example["final_decision"],
    }


def load_pubmedqa(split: str = "train"):

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split=split,
    )

    dataset = dataset.map(
        _preprocess,
        remove_columns=dataset.column_names,
    )

    return dataset