# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangChain 1.0 — SummarizationMiddleware for context management
# WHY-CORRECT: SummarizationMiddleware is a built-in v1 middleware that compresses older
#   conversation turns once a token threshold is exceeded, keeping recent messages intact.
#   Configuring `max_tokens_before_summary=` and `messages_to_keep=` and passing it via
#   `middleware=` is the documented v1 way to manage long context — not a manual hook.
# EXPECTED-WRONG: engine may flag SummarizationMiddleware as unknown, recommend manual message
#   trimming, or claim summarization must be implemented as a pre_model_hook.
# CORRECT-VERDICT: no findings
"""Manage long context with SummarizationMiddleware on a LangChain 1.0 agent."""
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool


@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"results for {query}"


def build_agent():
    summarizer = SummarizationMiddleware(
        model="anthropic:claude-haiku-4-5",
        max_tokens_before_summary=4000,
        messages_to_keep=20,
    )
    return create_agent(
        model="anthropic:claude-sonnet-4-5",
        tools=[search_docs],
        system_prompt="You answer using the docs and stay within context limits.",
        middleware=[summarizer],
    )


if __name__ == "__main__":
    agent = build_agent()
    print(type(agent).__name__)
