# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: OpenAI Python SDK (chat.completions, function/tool calling)
# WHY-CORRECT: The tool is declared with a precise JSON Schema (typed properties,
#   per-property descriptions, an enum-constrained unit, required fields, and
#   additionalProperties=false). The tool-result roundtrip is correct: the
#   assistant's tool_calls message is appended, then one tool-role message per
#   tool_call_id is appended before the follow-up request. Nothing here is
#   under-specified or missing.
# EXPECTED-WRONG: engine may suggest "add a JSON schema", "describe the tool
#   parameters", or "handle the tool result correctly" — all already done.
# CORRECT-VERDICT: no findings
"""Correct OpenAI function-calling roundtrip with a precise tool schema."""

import json

from openai import OpenAI

client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get the current weather for a city. Call this whenever the user "
                "asks about present weather conditions for a named location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country, e.g. 'Paris, France'.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit to report.",
                    },
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
        },
    }
]


def get_current_weather(location: str, unit: str) -> dict:
    """Look up the current weather (stubbed deterministic response)."""
    return {"location": location, "unit": unit, "temperature": 21, "summary": "sunny"}


def ask_weather(user_message: str) -> str:
    """Run one tool-calling turn and return the model's final text answer."""
    messages = [{"role": "user", "content": user_message}]

    first = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
    )
    choice = first.choices[0].message
    if not choice.tool_calls:
        return choice.content or ""

    # Append the assistant turn that requested the tool calls.
    messages.append(choice)

    # Execute each tool call and append one tool-role result per id.
    for call in choice.tool_calls:
        args = json.loads(call.function.arguments)
        result = get_current_weather(args["location"], args["unit"])
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            }
        )

    second = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
    )
    return second.choices[0].message.content or ""


if __name__ == "__main__":
    print(ask_weather("What's the weather in Paris right now?"))
