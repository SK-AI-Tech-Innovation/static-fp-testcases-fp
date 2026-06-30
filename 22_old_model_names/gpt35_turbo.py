# ACE-FP-EXPECT: clean
# CATEGORY: 22_old_model_names
# SOURCE: gpt-3.5-turbo + openai-python
# WHY-CORRECT: gpt-3.5-turbo is a valid, widely used, cheap chat model; the chat.completions call shape is correct and current.
# EXPECTED-WRONG: engine flags "use a newer model (gpt-4o / o-series)" as a best-practice finding, but model choice is not a defined best-practice category.
# CORRECT-VERDICT: no findings
"""Cheap single-turn chat completion using gpt-3.5-turbo."""

from openai import OpenAI

client = OpenAI()


def summarize(text: str) -> str:
    """Return a one-sentence summary of the supplied text."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You write concise one-sentence summaries."},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print(summarize("The quick brown fox jumps over the lazy dog repeatedly."))
