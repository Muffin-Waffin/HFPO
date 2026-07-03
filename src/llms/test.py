import time
from src.llms.loader import load_model

model, tokenizer = load_model()

messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=True, return_dict=True,
                                         return_tensors="pt").to(model.device)

start = time.perf_counter()
output = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id,
                          eos_token_id=tokenizer.eos_token_id)
print(f"Took {time.perf_counter() - start:.1f}s")
print(tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True))