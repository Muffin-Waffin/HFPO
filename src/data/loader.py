from .medqa import load_medqa
from .medmcqa import load_medmcqa
from .pubmedqa import load_pubmedqa


def load_dataset(name: str, split="train"):

    name = name.lower()

    if name == "medqa":
        return load_medqa(split)

    elif name == "medmcqa":
        return load_medmcqa(split)

    elif name == "pubmedqa":
        return load_pubmedqa(split)

    raise ValueError(f"Unknown dataset: {name}")