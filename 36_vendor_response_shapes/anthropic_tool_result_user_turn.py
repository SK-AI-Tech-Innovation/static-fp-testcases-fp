# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: Anthropic tool use multi-turn, verified June 2026
# WHY-CORRECT: Anthropic returns a tool result by appending a user-role turn whose content has a tool_result block keyed by tool_use_id; it is NOT OpenAI's {"role":"tool","tool_call_id":...}
# EXPECTED-WRONG: stale analyzer expects a "tool" role message with tool_call_id and flags the user-turn tool_result block as malformed
# CORRECT-VERDICT: no findings
"""Return a tool result to Anthropic as a tool_result block inside a user turn."""

import os

import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def main() -> None:
    messages = [{"role": "user", "content": "What's the weather in Seoul?"}]

    first = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

    tool_use = next(b for b in first.content if b.type == "tool_use")
    messages.append({"role": "assistant", "content": first.content})

    # tool_result is delivered in a USER turn, keyed by tool_use_id.
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": "18C and sunny",
                }
            ],
        }
    )

    final = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )
    print("".join(b.text for b in final.content if b.type == "text"))


if __name__ == "__main__":
    main()
