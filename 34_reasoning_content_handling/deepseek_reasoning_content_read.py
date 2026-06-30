# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: deepseek-reasoner (DeepSeek V3.2 thinking) + openai-python (DeepSeek OpenAI-compatible endpoint)
# WHY-CORRECT: deepseek-reasoner returns message.reasoning_content as a sibling field next to message.content;
#              reading both fields separately is the documented DeepSeek idiom for thinking models.
# EXPECTED-WRONG: engine flags message.reasoning_content as an invalid/nonexistent attribute on the chat completion
#                 message because an OpenAI-centric schema only knows message.content.
# CORRECT-VERDICT: no findings
"""Read DeepSeek reasoning_content and content as separate sibling fields."""

from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="${DEEPSEEK_API_KEY}",
)


def solve(problem: str) -> tuple[str, str]:
    """Return (chain_of_thought, final_answer) from deepseek-reasoner.

    Args:
        problem: The natural-language problem to reason about.

    Returns:
        A tuple of (reasoning_content, content).
    """
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": problem}],
    )

    message = response.choices[0].message
    # reasoning_content is a sibling of content on DeepSeek thinking models.
    chain_of_thought = message.reasoning_content
    final_answer = message.content
    return chain_of_thought, final_answer


if __name__ == "__main__":
    cot, answer = solve("Is 2027 a prime number? Show your reasoning.")
    print(answer)
