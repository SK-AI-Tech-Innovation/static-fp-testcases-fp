# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: vLLM offline batched inference (LLM + SamplingParams)
# WHY-CORRECT: vLLM's offline LLM engine loads weights locally and runs batched generate over a list
#              of prompts with explicit SamplingParams. This is the documented offline API; it does
#              not start a server and makes no HTTP calls. Batch + sampling config are complete.
# EXPECTED-WRONG: engine confuses vLLM's offline LLM with the OpenAI-compatible vLLM *server* and
#                 flags missing base_url/api_key/retries, or treats llm.generate() as non-AI because
#                 it isn't client.chat.completions.create.
# CORRECT-VERDICT: no findings
"""Batched offline generation with the vLLM LLM engine."""
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)


def generate_batch(prompts: list[str]) -> list[str]:
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]


if __name__ == "__main__":
    questions = [
        "What is the boiling point of water at sea level?",
        "Name the largest planet in the solar system.",
    ]
    for answer in generate_batch(questions):
        print(answer)
