# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: deepseek-reasoner (DeepSeek V3.2 thinking) + openai-python, tool-calling turn
# WHY-CORRECT: DeepSeek V3.2 thinking REQUIRES reasoning_content to be re-sent on any assistant turn that carries
#              tool_calls; omitting it returns HTTP 400 "Missing reasoning_content". Echoing it back is mandatory.
# EXPECTED-WRONG: engine flags including reasoning_content in the appended assistant message as sending an
#                 unknown/invalid field that "should be stripped", inverting the actual provider requirement.
# CORRECT-VERDICT: no findings
"""Re-send DeepSeek reasoning_content on the assistant tool-call turn (required)."""

import json

from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="${DEEPSEEK_API_KEY}",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def run(question: str) -> str:
    """Drive one tool-call round trip, re-sending reasoning_content as required.

    Args:
        question: The user question that may trigger a tool call.

    Returns:
        The final assistant content after the tool result is provided.
    """
    messages = [{"role": "user", "content": question}]

    first = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        tools=TOOLS,
    )
    msg = first.choices[0].message

    # DeepSeek V3.2 thinking: the assistant turn carrying tool_calls MUST also
    # carry reasoning_content back, or the next request returns 400.
    assistant_turn = {
        "role": "assistant",
        "content": msg.content,
        "reasoning_content": msg.reasoning_content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ],
    }
    messages.append(assistant_turn)

    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"Sunny, 22C in {args['city']}",
            }
        )

    second = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        tools=TOOLS,
    )
    return second.choices[0].message.content


if __name__ == "__main__":
    print(run("What's the weather in Seoul?"))
