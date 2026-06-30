# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: HuggingFace transformers high-level pipeline
# WHY-CORRECT: local text-generation pipeline loads a Llama checkpoint and runs inference fully
#              offline. No hosted API is involved, so hosted-API concerns (retries, structured
#              output, rate limits) are irrelevant. The code is complete and idiomatic.
# EXPECTED-WRONG: engine fails to recognize transformers.pipeline as LLM usage and flags it as
#                 non-AI, or applies API-shaped findings (e.g. "no retry/timeout on the LLM call")
#                 that make no sense for a local in-process generate.
# CORRECT-VERDICT: no findings
"""Generate text locally with a HuggingFace transformers pipeline."""
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)


def complete(prompt: str, max_new_tokens: int = 256) -> str:
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return outputs[0]["generated_text"]


if __name__ == "__main__":
    print(complete("Explain photosynthesis in one sentence:"))
