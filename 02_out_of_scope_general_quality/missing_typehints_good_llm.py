# ACE-FP-EXPECT: scoped-out
# CATEGORY: 02_out_of_scope_general_quality
# SOURCE: a correct tool-using agent loop wrapper written entirely without type hints
# WHY-CORRECT: the agent wrapper is correct — it runs a proper tool-use loop (call model, dispatch
#              tool_use blocks, feed tool_result back, repeat until end_turn) with a stop condition and
#              max_tokens. The only deficiency is missing type annotations, which the repo's own ruff ANN
#              rules cover — i.e. it's a typing/linting concern, explicitly out-of-scope for static here.
# EXPECTED-WRONG: "missing type hints / annotate parameters and return types (ANN001/ANN201)" findings.
# CORRECT-VERDICT: no findings (typing is out-of-scope; the agent loop pattern is already correct)
"""A correct Anthropic tool-use agent loop, deliberately written with no type hints."""
from __future__ import annotations

import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current temperature for a city in Celsius.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def run_tool(name, args):
    if name == "get_weather":
        return f"22C in {args['city']}"
    return "unknown tool"


def run_agent(user_message):
    messages = [{"role": "user", "content": user_message}]
    for _ in range(6):  # bounded loop — won't spin forever
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text").strip()
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
        messages.append({"role": "user", "content": tool_results})
    return "stopped: max turns reached"


if __name__ == "__main__":
    print(run_agent("What's the weather in Seoul?"))
