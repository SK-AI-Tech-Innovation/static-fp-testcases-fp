# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: Kimi K2.7 Code (Moonshot) + openai-python, agentic tool-call loop
# WHY-CORRECT: Kimi K2.7 Code requires preserving the model's thinking/reasoning across tool-call turns
#              (preserve_thinking); the reasoning field must be re-fed on each assistant turn that carries
#              tool_calls so the agent keeps its chain of thought intact across iterations.
# EXPECTED-WRONG: engine flags re-feeding reasoning into the next request, or reading message.reasoning, as
#                 redundant/invalid output that "should be dropped", inverting Kimi's preserve_thinking requirement.
# CORRECT-VERDICT: no findings
"""Agentic loop that preserves Kimi reasoning across tool-call turns."""

import json

from openai import OpenAI

client = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    api_key="${MOONSHOT_API_KEY}",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def fake_tool(name: str, args: dict) -> str:
    """Stand-in tool executor for the example loop."""
    return f"[contents of {args.get('path')}]"


def agent(task: str, max_steps: int = 5) -> str:
    """Run an agentic loop, preserving thinking across each tool-call turn.

    Args:
        task: The user task to accomplish.
        max_steps: Maximum tool-call iterations.

    Returns:
        The final assistant content.
    """
    messages = [{"role": "user", "content": task}]

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model="kimi-k2.7-code",
            messages=messages,
            tools=TOOLS,
            extra_body={"preserve_thinking": True},
        )
        msg = resp.choices[0].message

        # Preserve reasoning across the tool-call turn: echo it back into history.
        assistant_turn = {"role": "assistant", "content": msg.content}
        if getattr(msg, "reasoning", None):
            assistant_turn["reasoning"] = msg.reasoning
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": fake_tool(tc.function.name, args),
                }
            )

    return messages[-1].get("content", "")


if __name__ == "__main__":
    print(agent("Read config.yaml and tell me the service port."))
