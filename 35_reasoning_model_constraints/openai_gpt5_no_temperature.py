# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: gpt-5.5 + openai-python (Responses API)
# WHY-CORRECT: OpenAI reasoning models (GPT-5.x) reject sampling params. This Responses call deliberately omits temperature/top_p; adding them returns a 400 "unsupported parameter" error.
# EXPECTED-WRONG: stale analyzer flags "missing temperature, set temperature=0 for determinism"; applying that fix would 400 the request.
# CORRECT-VERDICT: no findings
"""Deterministic-style call to gpt-5.5 via the Responses API, with no sampling params (correct for reasoning models)."""

from openai import OpenAI

client = OpenAI()


def summarize(text: str) -> str:
    """Summarize text with gpt-5.5; reasoning models ignore/reject temperature, so we omit it."""
    response = client.responses.create(
        model="gpt-5.5",
        instructions="You are a concise technical summarizer.",
        input=text,
        max_output_tokens=512,
    )
    return response.output_text


if __name__ == "__main__":
    print(summarize("ACE is a static analyzer for LLM application code."))
