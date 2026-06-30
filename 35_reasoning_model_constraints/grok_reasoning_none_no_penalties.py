# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: grok-4.3 + openai-python (xAI OpenAI-compatible endpoint)
# WHY-CORRECT: xAI Grok reasoning models reject frequency/presence penalties and stop sequences, and expose reasoning_effort ("none"|"low"|"high"). This call sets reasoning_effort="none" and omits penalties/stop, the correct usage.
# EXPECTED-WRONG: stale analyzer flags "missing temperature/penalties, set temperature=0 and add frequency_penalty"; on Grok reasoning, penalties (and stop) 400 the request.
# CORRECT-VERDICT: no findings
"""grok-4.3 reasoning call with reasoning_effort="none" and no penalty/stop params (correct for Grok)."""

from openai import OpenAI

client = OpenAI(base_url="https://api.x.ai/v1", api_key="xai-...")


def quick_answer(question: str) -> str:
    """Fast answer from grok-4.3 with reasoning effort disabled; no penalties or stop sequences."""
    response = client.chat.completions.create(
        model="grok-4.3",
        messages=[{"role": "user", "content": question}],
        reasoning_effort="none",
        max_tokens=512,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(quick_answer("What is the capital of Australia?"))
