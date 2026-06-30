# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: Qwen3 thinking model run locally (transformers) — manual </think> token split
# WHY-CORRECT: when running Qwen3 locally the chain of thought is delimited by the </think> special token whose id
#              is 151668; splitting the generated id sequence at that token to separate thinking from the final
#              answer is the documented local-parsing idiom for Qwen3.
# EXPECTED-WRONG: engine flags the magic token id 151668 / manual </think> split as a hardcoded/invalid constant or
#                 "fragile string parsing" instead of recognizing the documented Qwen3 thinking delimiter.
# CORRECT-VERDICT: no findings
"""Split local Qwen3 output on the </think> token id 151668."""

from transformers import AutoModelForCausalLM, AutoTokenizer

THINK_END_TOKEN_ID = 151668  # </think> for Qwen3 tokenizers


def split_thinking(model_name: str, prompt: str) -> tuple[str, str]:
    """Generate locally and split thinking from the final answer at </think>.

    Args:
        model_name: A local Qwen3 thinking model identifier.
        prompt: The user prompt.

    Returns:
        A tuple of (thinking_text, answer_text).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=2048)

    output_ids = generated[0][len(inputs.input_ids[0]):].tolist()

    # Find the </think> delimiter (token id 151668) and split around it.
    try:
        split_at = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        split_at = 0

    thinking_text = tokenizer.decode(output_ids[:split_at], skip_special_tokens=True).strip()
    answer_text = tokenizer.decode(output_ids[split_at:], skip_special_tokens=True).strip()
    return thinking_text, answer_text


if __name__ == "__main__":
    thoughts, answer = split_thinking("Qwen/Qwen3-8B", "What is the 7th Fibonacci number?")
    print(answer)
