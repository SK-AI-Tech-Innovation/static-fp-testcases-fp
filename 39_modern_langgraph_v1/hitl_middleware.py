# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangChain 1.0 — HumanInTheLoopMiddleware(interrupt_on={...})
# WHY-CORRECT: In LangChain 1.0, agent behavior is composed with middleware classes rather than
#   pre/post hook callbacks. HumanInTheLoopMiddleware gates tool execution via the
#   `interrupt_on=` mapping (tool name -> approval config). It is passed through `middleware=`
#   on create_agent. This is the documented v1 HITL pattern.
# EXPECTED-WRONG: engine may flag HumanInTheLoopMiddleware as unknown, claim hooks like
#   `pre_model_hook`/`post_model_hook` are required, or that `interrupt_on` is not a valid kwarg.
# CORRECT-VERDICT: no findings
"""Gate tool calls for human approval using HumanInTheLoopMiddleware in a v1 agent."""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool


@tool
def transfer_funds(account: str, amount: int) -> str:
    """Transfer the given amount to an account."""
    return f"Transferred {amount} to {account}."


def build_agent():
    hitl = HumanInTheLoopMiddleware(
        interrupt_on={
            "transfer_funds": {"allowed_decisions": ["approve", "reject", "edit"]},
        },
    )
    return create_agent(
        model="anthropic:claude-sonnet-4-5",
        tools=[transfer_funds],
        system_prompt="You move money only after human approval.",
        middleware=[hitl],
    )


if __name__ == "__main__":
    agent = build_agent()
    print(type(agent).__name__)
