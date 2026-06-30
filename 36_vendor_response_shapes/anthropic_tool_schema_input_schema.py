# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: Anthropic tool use, verified June 2026
# WHY-CORRECT: Anthropic tool schema keys are {"name","description","input_schema"} passed via tools=[...]; it is NOT OpenAI's {"type":"function","function":{"parameters":...}}
# EXPECTED-WRONG: stale analyzer expects "parameters"/"function" wrapper and flags "input_schema" as a malformed/unknown tool field
# CORRECT-VERDICT: no findings
"""Define an Anthropic tool using the input_schema key and read tool_use output."""

import os

import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


def main() -> None:
    resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        tools=[WEATHER_TOOL],
        messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
    )

    for block in resp.content:
        if block.type == "tool_use":
            print(block.name, block.input)


if __name__ == "__main__":
    main()
