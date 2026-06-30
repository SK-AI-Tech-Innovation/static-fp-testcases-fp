# ACE-FP-EXPECT: clean
# CATEGORY: 31_prompt_caching_and_context
# SOURCE: OpenAI Python SDK (`openai`) message ordering for prefix cache reuse
# WHY-CORRECT: the cacheable, invariant content (system rules + few-shot examples) is built once
#              and always placed at the front of the message list; only the per-request user turn
#              is appended at the end. Keeping the prefix byte-stable maximizes prefix cache hits.
#              This is a correct optimization, not a missing one.
# EXPECTED-WRONG: dated skill pack flags "examples should move into a tool / fine-tune" or
#                 "non-deterministic prompt", missing that stable-prefix ordering is the goal.
# CORRECT-VERDICT: no findings
"""Order messages so the cacheable prefix stays stable across requests."""
from openai import OpenAI

client = OpenAI()

# Invariant prefix: built once, never reordered, reused for every call.
_STABLE_PREFIX = [
    {"role": "system", "content": "Classify the sentiment as positive, negative, or neutral."},
    {"role": "user", "content": "I love this!"},
    {"role": "assistant", "content": "positive"},
    {"role": "user", "content": "This is terrible."},
    {"role": "assistant", "content": "negative"},
]


def classify(text: str) -> str:
    messages = _STABLE_PREFIX + [{"role": "user", "content": text}]
    response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
    return response.choices[0].message.content


if __name__ == "__main__":
    print(classify("It was okay, nothing special."))
