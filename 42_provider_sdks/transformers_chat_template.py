# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: Hugging Face transformers text generation, verified June 2026
# WHY-CORRECT: local chat inference uses tokenizer.apply_chat_template(messages, add_generation_prompt=True) then model.generate; the output is decoded text, not an API response object
# EXPECTED-WRONG: stale analyzer expects response.choices[0].message.content and flags apply_chat_template + generate + tokenizer.decode as "no .choices / wrong response access"
# CORRECT-VERDICT: no findings
"""Generate text locally with transformers using a tokenizer chat template."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B-Instruct"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me one tip for learning Python."},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(input_ids, max_new_tokens=128)
    reply = tokenizer.decode(
        output_ids[0, input_ids.shape[-1] :], skip_special_tokens=True
    )
    print(reply)


if __name__ == "__main__":
    main()
