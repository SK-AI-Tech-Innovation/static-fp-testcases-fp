# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: generic reasoning-model integration (DeepSeek/Qwen-style reasoning_content) + openai-python
# WHY-CORRECT: storing the reasoning trace separately from the final answer (e.g. logging CoT to a debug sink while
#              returning only content to the user) is a deliberate, correct design; the reasoning is captured, not
#              lost, and is simply kept out of the user-facing channel.
# EXPECTED-WRONG: engine flags not returning/displaying reasoning_content alongside the answer as "dropping model
#                 output is data loss", missing that it is intentionally persisted to a separate store.
# CORRECT-VERDICT: no findings
"""Store reasoning separately from the final answer (intentional, not data loss)."""

import logging

from openai import OpenAI

logger = logging.getLogger("reasoning_trace")

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="${DEEPSEEK_API_KEY}",
)


def answer(problem: str, trace_store: dict[str, str]) -> str:
    """Return only the final answer; persist reasoning separately.

    Args:
        problem: The problem to solve.
        trace_store: A side store that captures the reasoning trace.

    Returns:
        The user-facing final answer (content only).
    """
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": problem}],
    )
    message = response.choices[0].message

    # Reasoning is captured to a separate sink, not discarded, and deliberately
    # kept out of the user-facing return value.
    if message.reasoning_content:
        trace_store["last_reasoning"] = message.reasoning_content
        logger.debug("captured %d chars of reasoning", len(message.reasoning_content))

    return message.content


if __name__ == "__main__":
    store: dict[str, str] = {}
    result = answer("What is the integral of 2x?", store)
    print(result)
