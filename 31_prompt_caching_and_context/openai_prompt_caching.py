# ACE-FP-EXPECT: clean
# CATEGORY: 31_prompt_caching_and_context
# SOURCE: OpenAI Python SDK (`openai`) Chat Completions with automatic prompt caching
# WHY-CORRECT: OpenAI caches prompt prefixes automatically (no API flag needed) for prompts
#              over the threshold, as long as the prefix is byte-stable. Here the long system
#              instructions are emitted first and unchanged every call, with only the user turn
#              varying at the end — the correct way to benefit from automatic caching.
# EXPECTED-WRONG: dated skill pack invents "no caching configured" or "add a cache_control
#                 parameter", not knowing OpenAI caching is implicit and prefix-based.
# CORRECT-VERDICT: no findings
"""Chat with a long, stable system prefix to benefit from OpenAI automatic prompt caching."""
from openai import OpenAI

client = OpenAI()

# Long, stable prefix (>1024 tokens) so the automatic prefix cache kicks in.
SYSTEM_PROMPT = "You are a precise code reviewer.\n" + ("Guideline line.\n" * 2000)


def review(snippet: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": snippet},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(review("def f(): return 1/0"))
