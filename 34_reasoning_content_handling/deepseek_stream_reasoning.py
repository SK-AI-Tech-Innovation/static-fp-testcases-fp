# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: deepseek-reasoner (DeepSeek V3.2 thinking) + openai-python streaming
# WHY-CORRECT: in streaming mode DeepSeek emits the chain of thought in delta.reasoning_content first, then the
#              answer in delta.content; reading both delta fields is the documented streaming idiom.
# EXPECTED-WRONG: engine flags delta.reasoning_content as an invalid streaming-chunk attribute because an
#                 OpenAI-centric delta schema only models delta.content / delta.role / delta.tool_calls.
# CORRECT-VERDICT: no findings
"""Stream DeepSeek delta.reasoning_content before delta.content."""

from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="${DEEPSEEK_API_KEY}",
)


def stream_solve(problem: str) -> tuple[str, str]:
    """Stream a deepseek-reasoner answer, accumulating reasoning then content.

    Args:
        problem: The problem to stream a solution for.

    Returns:
        A tuple of (reasoning_text, answer_text).
    """
    stream = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": problem}],
        stream=True,
    )

    reasoning_text = ""
    answer_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        # reasoning_content streams first; content follows once thinking ends.
        if getattr(delta, "reasoning_content", None):
            reasoning_text += delta.reasoning_content
            print(delta.reasoning_content, end="", flush=True)
        elif delta.content:
            answer_text += delta.content
            print(delta.content, end="", flush=True)

    return reasoning_text, answer_text


if __name__ == "__main__":
    stream_solve("Walk through solving x^2 - 5x + 6 = 0 step by step.")
