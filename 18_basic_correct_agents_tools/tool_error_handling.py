# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: Anthropic Python SDK (tool_result with is_error semantics)
# WHY-CORRECT: The tool catches its own failure modes and returns a structured
#   tool_result with is_error=True and an informative message the model can act
#   on, instead of crashing the loop or returning a bare string. This is the
#   documented good tool-error semantics — the failure is communicated back to
#   the model, not swallowed.
# EXPECTED-WRONG: engine may suggest "handle tool errors", "return errors to the
#   model", or "add a try/except" — error handling is already structured and
#   surfaced via is_error.
# CORRECT-VERDICT: no findings
"""A tool that returns a structured error result (good tool error semantics)."""

import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "divide",
        "description": "Divide the numerator by the denominator and return the quotient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "numerator": {"type": "number", "description": "The dividend."},
                "denominator": {"type": "number", "description": "The divisor."},
            },
            "required": ["numerator", "denominator"],
        },
    }
]


def run_divide(tool_input: dict) -> dict:
    """Execute the divide tool, returning a structured tool_result block."""
    try:
        quotient = tool_input["numerator"] / tool_input["denominator"]
    except ZeroDivisionError:
        return {
            "content": "Error: division by zero. Provide a non-zero denominator.",
            "is_error": True,
        }
    return {"content": str(quotient), "is_error": False}


def calculate(question: str) -> str:
    """Run one tool turn; tool failures come back to the model as errors."""
    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        tools=TOOLS,
        messages=messages,
    )

    if response.stop_reason != "tool_use":
        return next((b.text for b in response.content if b.type == "text"), "")

    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            outcome = run_divide(block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": outcome["content"],
                    "is_error": outcome["is_error"],
                }
            )
    messages.append({"role": "user", "content": tool_results})

    final = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        tools=TOOLS,
        messages=messages,
    )
    return next((b.text for b in final.content if b.type == "text"), "")


if __name__ == "__main__":
    print(calculate("What is 10 divided by 0?"))
