from src.models.loader import load_model
from src.data.loader import load_dataset
from src.evaluation.scorer import score_sample


def main():

    model, tokenizer = load_model()

    dataset = load_dataset("medqa")

    sample = dataset[0]

    score, prediction, response = score_sample(
        model,
        tokenizer,
        """You are an expert medical assistant.

        Solve the question internally.

        After your reasoning, output ONLY ONE CAPITAL LETTER.

        Example:

        D

        Do not output any explanation after the letter.""",
        sample,
    )

    print("=" * 80)
    print("Ground Truth:", sample["answer"])
    print("Prediction :", prediction)
    print("Score      :", score)
    print()
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()