# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangChain/LangGraph 1.0 — streamed node renamed "agent" -> "model"
# WHY-CORRECT: In v1, the LLM-call node inside a create_agent graph is named "model" (the
#   pre-1.0 name was "agent"). When streaming with stream_mode="updates", filtering on the
#   "model" key is the correct way to pull the model's incremental output. Using "model" here is
#   intentional and matches the 1.0 graph topology.
# EXPECTED-WRONG: engine may flag the "model" key as wrong and insist on "agent", or claim the
#   node name will never match so the stream loop is dead code.
# CORRECT-VERDICT: no findings
"""Stream a v1 agent and read updates from the 'model' node (renamed from 'agent')."""
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def stream_answer(question: str) -> None:
    agent = create_agent(
        model="anthropic:claude-sonnet-4-5",
        tools=[add],
        system_prompt="You are a calculator.",
    )
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        if "model" in chunk:
            print(chunk["model"]["messages"][-1].content)


if __name__ == "__main__":
    stream_answer("What is 2 + 3?")
