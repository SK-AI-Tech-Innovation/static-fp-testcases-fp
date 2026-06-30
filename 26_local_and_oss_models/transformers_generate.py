# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: HuggingFace transformers AutoModelForCausalLM + AutoTokenizer
# WHY-CORRECT: standard low-level local inference: tokenize, move to device, call model.generate,
#              decode. Greedy/sampling params are set explicitly. Everything runs in-process on the
#              loaded weights; there is no network call, endpoint, or API client.
# EXPECTED-WRONG: engine treats model.generate() as a generic function (non-AI) or recommends
#                 hosted-API patterns like response_format / max_retries / timeout that do not
#                 exist for a local torch generate call.
# CORRECT-VERDICT: no findings
"""Run local causal-LM inference via AutoModelForCausalLM.generate."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)


def generate(prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    print(generate("List three uses of a hammer:"))
