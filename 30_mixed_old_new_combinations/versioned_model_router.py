# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python v1.x with a feature-flag router selecting an old vs new model id
# WHY-CORRECT: a single client.chat.completions.create call parameterized by a model id chosen from a feature flag is valid regardless of which model (old gpt-3.5-turbo or new gpt-4o) is selected; both are servable chat models. Routing by flag does not change API correctness.
# EXPECTED-WRONG: engine may flag the old model branch as deprecated or claim mixing model generations behind one call site is an error.
# CORRECT-VERDICT: no findings
"""Route requests to an old or new model based on a feature flag."""

import os

from openai import OpenAI

client = OpenAI()

OLD_MODEL = "gpt-3.5-turbo"
NEW_MODEL = "gpt-4o"


def pick_model() -> str:
    if os.environ.get("USE_NEW_MODEL", "false").lower() == "true":
        return NEW_MODEL
    return OLD_MODEL


def ask(prompt: str) -> str:
    response = client.chat.completions.create(
        model=pick_model(),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask("Give me a fun fact about octopuses."))
