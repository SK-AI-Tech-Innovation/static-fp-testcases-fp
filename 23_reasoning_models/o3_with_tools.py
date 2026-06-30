# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: o3 + openai-python (function/tool calling)
# WHY-CORRECT: o3 supports tool calling; the tools schema, tool_calls handling, and tool-result round-trip are all correctly formed, with max_completion_tokens.
# EXPECTED-WRONG: engine flags tool use on a reasoning model as unsupported, or flags max_completion_tokens, but reasoning models do support tools and this param.
# CORRECT-VERDICT: no findings
"""Tool calling with the o3 reasoning model, done correctly."""

import json

from openai import OpenAI

client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def get_population(city: str) -> int:
    """Stub data source."""
    return {"Tokyo": 13960000}.get(city, 0)


def run(question: str) -> str:
    """Answer a question, letting o3 call get_population as needed."""
    messages = [{"role": "user", "content": question}]
    response = client.chat.completions.create(
        model="o3",
        messages=messages,
        tools=TOOLS,
        max_completion_tokens=4096,
    )
    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content

    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = get_population(args["city"])
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            }
        )

    follow_up = client.chat.completions.create(
        model="o3",
        messages=messages,
        tools=TOOLS,
        max_completion_tokens=4096,
    )
    return follow_up.choices[0].message.content


if __name__ == "__main__":
    print(run("What is the population of Tokyo?"))
