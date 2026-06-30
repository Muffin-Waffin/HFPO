from src.llms.loader import load_model
from src.data.loader import load_dataset
from src.evaluation.scorer import score_prompt
from src.prompts.baseline import BASELINE_PROMPTS
import csv

def main():

    print("=" * 80)
    print("Loading model...")
    model, tokenizer = load_model()

    print("Loading dataset...")
    dataset = load_dataset("medqa")

    # Use a small subset while developing.
    dataset = dataset.select(range(20))

    print(f"Evaluating on {len(dataset)} samples.\n")

    results = []

    for name, prompt in BASELINE_PROMPTS.items():

        print("=" * 80)
        print(f"Prompt: {name}")

        accuracy = score_prompt(
            model=model,
            tokenizer=tokenizer,
            system_prompt=prompt,
            dataset=dataset,
        )

        results.append({
        "prompt": name,
        "accuracy": accuracy,
        })

        print(f"Accuracy: {accuracy:.4f}")



    print("=" * 80)
    print("Baseline evaluation complete.")

    with open(
        "results/baseline_results.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["prompt", "accuracy"],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to results/baseline_results.csv")


if __name__ == "__main__":
    main()