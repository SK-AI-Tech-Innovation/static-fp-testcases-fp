# ACE-FP-EXPECT: clean
# CATEGORY: 22_old_model_names
# SOURCE: gpt-4-0613 + openai-python
# WHY-CORRECT: gpt-4-0613 is a deliberately pinned dated snapshot for reproducible output; pinning a snapshot is an intentional, correct practice.
# EXPECTED-WRONG: engine flags the dated snapshot / "old model" as something to upgrade, but a deliberate pin is a design choice, not a defect.
# CORRECT-VERDICT: no findings
"""Reproducible classification pinned to the gpt-4-0613 snapshot."""

from openai import OpenAI

client = OpenAI()

# Pinned snapshot on purpose: regression tests assert exact outputs, so the
# model version must not drift. Do not "upgrade" this without re-baselining.
MODEL_SNAPSHOT = "gpt-4-0613"


def classify_sentiment(review: str) -> str:
    """Classify a product review as positive, negative, or neutral."""
    response = client.chat.completions.create(
        model=MODEL_SNAPSHOT,
        messages=[
            {"role": "system", "content": "Reply with exactly one word: positive, negative, or neutral."},
            {"role": "user", "content": review},
        ],
        temperature=0,
        max_tokens=1,
    )
    return response.choices[0].message.content.strip().lower()


if __name__ == "__main__":
    print(classify_sentiment("This is the best purchase I've made all year."))
