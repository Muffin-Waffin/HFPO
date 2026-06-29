from datasets import load_dataset


def _preprocess(example):
    return {
        "question": example["question"],
        "context": "",
        "choices": [
            example["opa"],
            example["opb"],
            example["opc"],
            example["opd"],
        ],
        "answer": example["cop"],
    }


def load_medmcqa(split: str = "train"):

    dataset = load_dataset(
        "openlifescienceai/medmcqa",
        split=split,
    )

    dataset = dataset.map(
        _preprocess,
        remove_columns=dataset.column_names,
    )

    return dataset