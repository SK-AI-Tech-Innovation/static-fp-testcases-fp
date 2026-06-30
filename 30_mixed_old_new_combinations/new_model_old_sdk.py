# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python v0.28 (legacy openai.ChatCompletion.create) calling a NEW model gpt-4o
# WHY-CORRECT: the v0 ChatCompletion endpoint accepts any model string the account can serve; gpt-4o is a valid model id on the legacy /v1/chat/completions route. Old SDK + new model works as long as the model is available.
# EXPECTED-WRONG: engine may claim "unknown model" because gpt-4o postdates the v0 SDK, or push a v1 client migration as if it were required for correctness.
# CORRECT-VERDICT: no findings
"""Call the gpt-4o model through the legacy openai v0 ChatCompletion API."""

import openai

openai.api_key = "set-via-env-in-real-use"


def ask(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(ask("What is the boiling point of water at sea level?"))
