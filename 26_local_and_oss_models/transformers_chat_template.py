# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: HuggingFace transformers tokenizer.apply_chat_template + generate
# WHY-CORRECT: modern HF chat workflow. Messages are rendered through the model's own chat template
#              (add_generation_prompt=True), tokenized, and fed to model.generate. This is the
#              recommended way to do instruction-tuned local chat; roles/templating are correct.
# EXPECTED-WRONG: engine sees `messages=[{"role": ...}]` and assumes an OpenAI-style hosted call,
#                 then flags missing API params (model name string, retries, structured output),
#                 or fails to detect it as chat at all because there is no client.chat.completions.
# CORRECT-VERDICT: no findings
"""Local instruction-tuned chat using apply_chat_template then generate."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)


def chat(messages: list[dict], max_new_tokens: int = 512) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
    )
    reply_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True)


if __name__ == "__main__":
    conversation = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Give me a one-line tip for better sleep."},
    ]
    print(chat(conversation))
