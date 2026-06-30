# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: OpenAI Python SDK — Responses API (the current successor to the
#   deprecated Assistants API v2). OpenAI has announced Assistants API v2 is
#   being sunset in favor of the Responses API, so this models the current
#   equivalent: a stateful multi-turn thread via previous_response_id plus a
#   tool roundtrip.
# WHY-CORRECT: Uses the current Responses API correctly — a first response with
#   a precisely-described function tool, then a follow-up that chains state with
#   previous_response_id and submits a typed function_call_output keyed by
#   call_id. Stateful continuation and tool roundtrip are both handled the
#   documented way.
# EXPECTED-WRONG: engine may flag "Assistants API v2 is deprecated, migrate to
#   Responses" (already done) or "chain thread state" / "submit tool outputs"
#   (both present).
# CORRECT-VERDICT: no findings
"""OpenAI Responses API (current successor to Assistants v2): stateful tool run."""

import json

from openai import OpenAI

client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "name": "get_order_status",
        "description": (
            "Look up the shipping status of an order. Call this when the user "
            "asks where their order is or whether it has shipped."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier, e.g. 'A-10293'.",
                },
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    }
]


def get_order_status(order_id: str) -> dict:
    """Look up an order's status (stubbed deterministic response)."""
    return {"order_id": order_id, "status": "shipped", "eta_days": 2}


def ask_order(user_message: str) -> str:
    """Run a stateful tool turn over the Responses API and return the answer."""
    first = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": user_message}],
        tools=TOOLS,
    )

    # Collect any function calls the model requested this turn.
    function_calls = [item for item in first.output if item.type == "function_call"]
    if not function_calls:
        return first.output_text

    # Submit one typed function_call_output per call_id, chaining thread state.
    tool_outputs = []
    for call in function_calls:
        args = json.loads(call.arguments)
        result = get_order_status(args["order_id"])
        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            }
        )

    second = client.responses.create(
        model="gpt-4o",
        previous_response_id=first.id,
        input=tool_outputs,
        tools=TOOLS,
    )
    return second.output_text


if __name__ == "__main__":
    print(ask_order("Where is my order A-10293?"))
