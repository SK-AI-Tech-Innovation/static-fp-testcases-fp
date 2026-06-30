# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: Anthropic Python SDK (Messages API, multiple tool_use blocks in one turn)
# WHY-CORRECT: When the model emits several tool_use blocks in a single
#   assistant turn, the loop executes each one and collects ALL tool_result
#   blocks (each keyed by its own tool_use_id) into a single user message before
#   the follow-up request. This is exactly the documented way to handle parallel
#   tool calls — no result is dropped or mis-paired.
# EXPECTED-WRONG: engine may suggest "handle multiple tool calls" or "match each
#   result to its tool_use_id" — both already done correctly.
# CORRECT-VERDICT: no findings
"""Correctly handling multiple parallel tool calls in one turn."""

import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_temperature",
        "description": "Get the current temperature in Celsius for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
            },
            "required": ["city"],
        },
    }
]

_TEMPS = {"Paris": 21, "Tokyo": 26, "Oslo": 9}


def get_temperature(city: str) -> str:
    """Return the stubbed current temperature for a city."""
    return f"{_TEMPS.get(city, 15)}C"


def compare_cities(question: str) -> str:
    """Ask a question that triggers parallel tool calls and return the answer."""
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

    # Execute every tool_use block from this turn; collect all results.
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": get_temperature(block.input["city"]),
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
    print(compare_cities("Which is warmer right now, Paris or Tokyo?"))
