# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: LangGraph v1 prebuilt agent via `create_agent(..., system_prompt=...)`.
# WHY-CORRECT: in LangGraph v1, `create_agent` is the current prebuilt-agent constructor and it takes
#              `system_prompt=`. This supersedes the older `create_react_agent(prompt=...)`. The call is the
#              correct modern API.
# EXPECTED-WRONG: a stale engine "knows" the prebuilt agent as `create_react_agent` with a `prompt=` kwarg
#                 and "fixes" the code by renaming `create_agent`→`create_react_agent` and
#                 `system_prompt=`→`prompt=`. That routes to the deprecated constructor / wrong keyword — a
#                 downgrade to an older API surface.
# CORRECT-VERDICT: no findings — keep `create_agent(system_prompt=...)`. Do not rewrite to
#                  `create_react_agent(prompt=...)`.
"""LangGraph v1 prebuilt agent via create_agent(system_prompt=...) — by design."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


agent = create_agent(
    model="anthropic:claude-opus-4-8",
    tools=[add],
    system_prompt="You are a careful arithmetic assistant. Use the tools provided.",
)


def run(question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
