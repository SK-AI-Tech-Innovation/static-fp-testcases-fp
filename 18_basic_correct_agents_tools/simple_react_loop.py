# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: Anthropic Python SDK (manual agentic loop, ReAct-style)
# WHY-CORRECT: The loop is explicitly bounded by max_steps, so it cannot run
#   forever. Each iteration appends the assistant content, executes any tool
#   calls, feeds tool_result blocks back keyed by tool_use_id, and breaks on
#   end_turn. One focused tool with a clear schema. Bounded + correct roundtrip.
# EXPECTED-WRONG: engine may suggest "bound the agent loop / add a max-iteration
#   limit" or "guard against infinite loops" — the loop is already bounded.
# CORRECT-VERDICT: no findings
"""A small, bounded ReAct-style loop that calls one tool correctly."""

import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "add",
        "description": "Add two integers and return their sum.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First addend."},
                "b": {"type": "integer", "description": "Second addend."},
            },
            "required": ["a", "b"],
        },
    }
]


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def run(question: str, max_steps: int = 5) -> str:
    """Run a bounded reason-act loop and return the final text answer."""
    messages = [{"role": "user", "content": question}]

    for _ in range(max_steps):
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if b.type == "text"), "")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = add(block.input["a"], block.input["b"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    # Bounded fallback: hit the step ceiling without a natural end_turn.
    return "Step limit reached before a final answer was produced."


if __name__ == "__main__":
    print(run("What is 17 + 25?"))
