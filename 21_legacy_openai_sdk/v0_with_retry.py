# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: openai-python v0.x (e.g. 0.28) — authentic legacy API
# WHY-CORRECT: v0.x did not auto-retry, so wrapping ChatCompletion.create in a manual
#   exponential-backoff loop on RateLimitError / APIError was the recommended reliability
#   pattern (mirrors OpenAI cookbook examples of the era). The retry logic itself is sound.
# EXPECTED-WRONG: engine may flag the v0 call as deprecated, claim "use the built-in max_retries
#   of the v1 client" (which did not exist in v0), or mis-fix the dict response access.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy openai v0.x chat call wrapped in manual exponential-backoff retries."""
import os
import random
import time

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def chat_with_retry(prompt: str, max_retries: int = 5) -> str:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response["choices"][0]["message"]["content"]
        except (openai.error.RateLimitError, openai.error.APIError,
                openai.error.Timeout) as exc:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.uniform(0, 1)
            print(f"Retryable error ({exc}); backing off {sleep_for:.1f}s")
            time.sleep(sleep_for)
            delay *= 2
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    print(chat_with_retry("Say hello."))
