# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: OpenAI Realtime API function calling (gpt-realtime-2) over WebSocket events
# WHY-CORRECT: in the Realtime protocol a tool call surfaces as the server event
#              `response.function_call_arguments.done`, and the reply is sent back by creating a
#              `conversation.item.create` item of type `function_call_output` — NOT by reading
#              `.choices[0].message.tool_calls`. This is the canonical, complete pattern.
# EXPECTED-WRONG: a chat-API-centric engine expects `.choices`/`tool_calls` parsing and flags
#                 "AI call missing structured output" or "tool call not handled correctly".
# CORRECT-VERDICT: no findings
"""Handle a Realtime function (tool) call and return its result as an event."""
from __future__ import annotations

import json

import websockets


async def configure_tools(ws: websockets.WebSocketClientProtocol) -> None:
    """Register a single weather tool via session.update."""
    await ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": "Get the current weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        }
                    ],
                    "tool_choice": "auto",
                },
            }
        )
    )


async def handle_events(ws: websockets.WebSocketClientProtocol) -> None:
    """React to function-call events and feed results back into the session."""
    async for raw in ws:
        event = json.loads(raw)
        if event.get("type") == "response.function_call_arguments.done":
            args = json.loads(event["arguments"])
            result = _lookup_weather(args["city"])
            # Reply to the tool call by inserting a function_call_output item,
            # then ask the model to continue speaking.
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": event["call_id"],
                            "output": json.dumps(result),
                        },
                    }
                )
            )
            await ws.send(json.dumps({"type": "response.create"}))


def _lookup_weather(city: str) -> dict[str, object]:
    """Stand-in weather lookup."""
    return {"city": city, "temp_c": 21, "sky": "clear"}
