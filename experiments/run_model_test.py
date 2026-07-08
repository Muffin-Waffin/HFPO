from src.llms.loader import load_model

import torch
from configs.config import MAX_NEW_TOKENS


def main():

    model, tokenizer = load_model()

    question = (
        "A 45-year-old patient with Type 2 Diabetes has "
        "HbA1c 8.5%. Metformin is insufficient. "
        "What is the recommended next treatment?"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical expert.\n"
                "Answer with ONLY the final answer.\n"
                "Do NOT explain your reasoning.\n"
                "Do NOT output <think> tags.\n"
                "Respond in one or two sentences."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs.input_ids.shape[-1]:]

    answer = tokenizer.decode(
    generated,
    skip_special_tokens=True,
    ).strip()

    # Remove empty reasoning tags
    answer = answer.replace("<think>\n</think>", "").strip()
    answer = answer.replace("<think></think>", "").strip()

    print("\nQuestion:\n")
    print(question)

    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()