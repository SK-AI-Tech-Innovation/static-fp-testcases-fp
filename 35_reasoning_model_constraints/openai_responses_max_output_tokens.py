# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: gpt-5.5 + openai-python (Responses API)
# WHY-CORRECT: The Responses API caps output with max_output_tokens; max_tokens is not a valid Responses field and max_completion_tokens belongs to Chat Completions. No temperature/top_p is sent, correct for reasoning models.
# EXPECTED-WRONG: stale analyzer flags "missing temperature, set temperature=0" or insists on max_tokens; either change errors on this Responses reasoning call.
# CORRECT-VERDICT: no findings
"""Responses API call that correctly uses max_output_tokens and omits sampling params."""

from openai import OpenAI

client = OpenAI()


def classify(ticket: str) -> str:
    """Classify a support ticket with the Responses API, bounding output via max_output_tokens."""
    response = client.responses.create(
        model="gpt-5.5",
        instructions="Classify the ticket as one of: billing, technical, account.",
        input=ticket,
        max_output_tokens=64,
    )
    return response.output_text.strip()


if __name__ == "__main__":
    print(classify("My card was charged twice for the same subscription."))
