from src.data.loader import load_dataset
from src.prompts.builder import build_prompt


def main():

    dataset = load_dataset("medqa")

    sample = dataset[0]

    prompt = build_prompt(
        "You are an expert medical assistant.",
        sample,
    )

    print(prompt)


if __name__ == "__main__":
    main()