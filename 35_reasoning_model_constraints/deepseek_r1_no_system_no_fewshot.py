# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: deepseek-r1 + openai-python (OpenAI-compatible endpoint)
# WHY-CORRECT: DeepSeek R1 is documented to perform best with no system prompt and zero-shot (no few-shot examples), with all guidance in a single user message. This call follows that guidance exactly.
# EXPECTED-WRONG: stale analyzer flags "missing system prompt, add a system message" and "missing few-shot examples"; both degrade R1 per the model card and the system role is discouraged.
# CORRECT-VERDICT: no findings
"""DeepSeek R1 zero-shot call: a single user message, no system prompt, no few-shot examples."""

from openai import OpenAI

client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-...")


def solve_math(problem: str) -> str:
    """Solve a math problem with R1 using the recommended zero-shot, no-system pattern."""
    prompt = (
        f"{problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    response = client.chat.completions.create(
        model="deepseek-r1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(solve_math("Find all real x such that x^2 - 5x + 6 = 0."))
