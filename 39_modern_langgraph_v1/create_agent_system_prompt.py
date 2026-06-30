# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangChain 1.0 — `from langchain.agents import create_agent`
# WHY-CORRECT: In LangChain 1.0 the production agent constructor is `create_agent`, exported
#   from `langchain.agents`. It supersedes the pre-1.0 `create_react_agent` from
#   `langgraph.prebuilt`. The static-instruction parameter is `system_prompt=` and accepts a
#   plain string. Tools are passed as a list; the returned object is a compiled graph invoked
#   with `.invoke({"messages": [...]})`.
# EXPECTED-WRONG: engine may flag `create_agent` as unknown ("did you mean create_react_agent")
#   or claim the keyword should be `prompt=` rather than `system_prompt=`.
# CORRECT-VERDICT: no findings
"""Build a LangChain 1.0 agent with create_agent and a string system_prompt."""
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Return a canned weather string for the given city."""
    return f"It is sunny in {city}."


def build_agent():
    return create_agent(
        model="anthropic:claude-sonnet-4-5",
        tools=[get_weather],
        system_prompt="You are a concise weather assistant. Answer in one sentence.",
    )


if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris?"}]})
    print(result["messages"][-1].content)
