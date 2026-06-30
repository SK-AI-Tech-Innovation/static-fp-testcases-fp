# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: self-hosted vLLM OpenAI-compatible server with a reasoning parser + openai-python streaming
# WHY-CORRECT: vLLM's reasoning parser exposes the chain of thought as delta.reasoning (and message.reasoning),
#              NOT reasoning_content; reading delta.reasoning is the correct field for a vLLM-served model.
# EXPECTED-WRONG: engine flags delta.reasoning as an invalid attribute (expecting delta.content only), or "corrects"
#                 it to reasoning_content which would be wrong for vLLM specifically.
# CORRECT-VERDICT: no findings
"""Read vLLM delta.reasoning (not reasoning_content) while streaming."""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)


def stream_solve(problem: str) -> tuple[str, str]:
    """Stream from a vLLM server, reading delta.reasoning for the CoT.

    Args:
        problem: The problem to stream a solution for.

    Returns:
        A tuple of (reasoning_text, answer_text).
    """
    stream = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        messages=[{"role": "user", "content": problem}],
        stream=True,
    )

    reasoning_text = ""
    answer_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        # vLLM's reasoning parser uses 'reasoning', not 'reasoning_content'.
        if getattr(delta, "reasoning", None):
            reasoning_text += delta.reasoning
        elif delta.content:
            answer_text += delta.content

    return reasoning_text, answer_text


if __name__ == "__main__":
    cot, answer = stream_solve("Count the number of r's in 'strawberry'.")
    print(answer)
