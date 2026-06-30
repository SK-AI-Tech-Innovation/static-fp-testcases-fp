# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: DeepSeek V3.2 reasoner via the OpenAI-compatible endpoint, multi-turn tool call that preserves
#         `reasoning_content` on the assistant turn being replayed.
# WHY-CORRECT: on DeepSeek's reasoning models, when you continue a conversation that contains a tool call,
#              the assistant message you append back must carry its `reasoning_content` alongside
#              `tool_calls`. Dropping it returns `400 Missing reasoning_content`. This code correctly carries
#              `reasoning_content` through on the replayed assistant turn.
# EXPECTED-WRONG: a stale engine "knows" the OpenAI convention where assistant messages have only
#                 `content` / `tool_calls` and treats `reasoning_content` as a stray/unknown field — it
#                 "cleans up" the message by stripping `reasoning_content` before re-sending. On DeepSeek V3.2
#                 that strip causes `400 Missing reasoning_content`. The "fix" breaks the tool turn.
# CORRECT-VERDICT: no findings — keep `reasoning_content` on the replayed assistant message. Do not strip it.
"""DeepSeek V3.2 tool turn that preserves reasoning_content on replay — by design."""
from __future__ import annotations

import json
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current temperature for a city, Celsius.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def ask(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    first = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        tools=TOOLS,
    )
    msg = first.choices[0].message
    if not msg.tool_calls:
        return msg.content

    # Replay the assistant turn WITH reasoning_content — DeepSeek requires it on tool-call continuation.
    messages.append(
        {
            "role": "assistant",
            "content": msg.content,
            "reasoning_content": msg.reasoning_content,
            "tool_calls": msg.tool_calls,
        }
    )
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"city": args["city"], "celsius": 19.0}),
            }
        )

    final = client.chat.completions.create(
        model="deepseek-reasoner", messages=messages, tools=TOOLS
    )
    return final.choices[0].message.content
