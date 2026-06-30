# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: gpt-5 + openai-python (Responses API)
# WHY-CORRECT: the Responses API responses.create with reasoning={"effort": ...} and text={"verbosity": ...} is the correct gpt-5 reasoning idiom; output_text is the right accessor.
# EXPECTED-WRONG: engine flags reasoning/verbosity as unknown params, or flags use of responses.create instead of chat.completions, but these are correct for gpt-5.
# CORRECT-VERDICT: no findings
"""Responses-API reasoning request to gpt-5 with effort and verbosity."""

from openai import OpenAI

client = OpenAI()


def plan(task: str) -> str:
    """Produce a concise plan with gpt-5 via the Responses API."""
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "medium"},
        text={"verbosity": "low"},
        input=[
            {"role": "developer", "content": "Return a numbered plan, nothing else."},
            {"role": "user", "content": task},
        ],
    )
    return response.output_text


if __name__ == "__main__":
    print(plan("Migrate a monolith to a service-oriented architecture."))
