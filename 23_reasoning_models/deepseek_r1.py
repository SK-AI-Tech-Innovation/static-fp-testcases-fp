# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: deepseek-reasoner (DeepSeek-R1) + openai-python (DeepSeek OpenAI-compatible endpoint)
# WHY-CORRECT: DeepSeek-R1 exposes reasoning_content alongside content on the message; reading both separately is the correct idiom.
# EXPECTED-WRONG: engine flags reading message.reasoning_content as accessing a nonexistent attribute, but R1 populates it on its OpenAI-compatible responses.
# CORRECT-VERDICT: no findings
"""DeepSeek-R1 request reading reasoning_content and content separately."""

from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="${DEEPSEEK_API_KEY}",
)


def solve(problem: str) -> tuple[str, str]:
    """Return (chain_of_thought, final_answer) from deepseek-reasoner."""
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": problem}],
        max_tokens=4096,
    )
    message = response.choices[0].message
    # R1 separates its chain-of-thought into reasoning_content; the user-facing
    # answer is in content. Do not feed reasoning_content back into history.
    return message.reasoning_content, message.content


if __name__ == "__main__":
    cot, answer = solve("What is 17 * 23, computed step by step?")
    print(answer)
