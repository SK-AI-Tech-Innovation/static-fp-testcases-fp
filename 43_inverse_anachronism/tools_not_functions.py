# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: OpenAI Chat Completions function/tool calling via the current `tools=` / `tool_choice=` API.
# WHY-CORRECT: `tools=[{"type": "function", "function": {...}}]` plus `tool_choice=` is OpenAI's current
#              tool-calling surface; the response carries `message.tool_calls` and results are fed back with
#              `{"role": "tool", "tool_call_id": ...}`. This is the modern, correct shape.
# EXPECTED-WRONG: a stale engine "knows" the legacy `functions=[{...}]` + `function_call=` parameters and
#                 "fixes" the code by downgrading `tools`→`functions`, `tool_choice`→`function_call`,
#                 flattening the `{"type":"function","function":{...}}` wrapper, and reading
#                 `message.function_call` instead of `message.tool_calls`. The legacy `functions` API is
#                 deprecated and loses parallel tool calls — a strict downgrade.
# CORRECT-VERDICT: no findings — keep `tools=` / `tool_choice=` and `message.tool_calls`. Do not revert to
#                  `functions=` / `function_call=`.
"""OpenAI tool calling with the current tools= API — not legacy functions=, by design."""
from __future__ import annotations

import json
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Look up the latest price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    }
]


def _get_stock_price(symbol: str) -> dict:
    return {"symbol": symbol, "price": 187.42}


def ask(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    if not msg.tool_calls:
        return msg.content

    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = _get_stock_price(**args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            }
        )

    final = client.chat.completions.create(model="gpt-4.1", messages=messages, tools=TOOLS)
    return final.choices[0].message.content
