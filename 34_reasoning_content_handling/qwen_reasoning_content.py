# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: Qwen thinking model via Alibaba DashScope OpenAI-compatible endpoint + openai-python
# WHY-CORRECT: DashScope exposes Qwen's chain of thought in choices[0].message.reasoning_content, a sibling of
#              content; reading it directly is the documented DashScope idiom for Qwen thinking models.
# EXPECTED-WRONG: engine flags message.reasoning_content as a nonexistent attribute because an OpenAI-centric
#                 schema only knows message.content.
# CORRECT-VERDICT: no findings
"""Read Qwen reasoning_content from a DashScope OpenAI-compatible response."""

from openai import OpenAI

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="${DASHSCOPE_API_KEY}",
)


def solve(problem: str) -> tuple[str, str]:
    """Return (reasoning, answer) from a Qwen thinking model on DashScope.

    Args:
        problem: The problem to reason about.

    Returns:
        A tuple of (reasoning_content, content).
    """
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": problem}],
        extra_body={"enable_thinking": True},
    )

    message = response.choices[0].message
    reasoning = message.reasoning_content
    answer = message.content
    return reasoning, answer


if __name__ == "__main__":
    cot, answer = solve("If a train travels 60km in 45 minutes, what is its speed in km/h?")
    print(answer)
