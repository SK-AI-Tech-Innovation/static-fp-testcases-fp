# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: LangChain `get_openai_callback()` / custom callback handler recording token usage
# WHY-CORRECT: the callback context captures prompt tokens, completion tokens, and cost for
#              every invocation inside the block and a custom handler logs them — per-call
#              usage/cost observability is satisfied
# EXPECTED-WRONG: engine flags "no token usage tracking" because the create call has no inline
#                 usage handling; the accounting happens via the callback handler
# CORRECT-VERDICT: no findings
"""Track per-call token usage and cost via a LangChain callback handler."""
import logging
from typing import Any

from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

logger = logging.getLogger("llm.usage")
llm = ChatOpenAI(model="gpt-4.1-mini")


class UsageLogger(BaseCallbackHandler):
    """Emits prompt/completion token counts for each finished LLM generation."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage", {})
        logger.info(
            "llm_usage",
            extra={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        )


def summarize(text: str) -> str:
    with get_openai_callback() as cb:
        msg = llm.invoke(
            f"Summarize:\n{text}",
            config={"callbacks": [UsageLogger()]},
        )
        logger.info("call_cost", extra={"total_tokens": cb.total_tokens, "usd": cb.total_cost})
    return msg.content
