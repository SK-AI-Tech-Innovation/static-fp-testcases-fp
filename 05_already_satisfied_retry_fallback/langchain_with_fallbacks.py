# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: LangChain `Runnable.with_retry()` + `Runnable.with_fallbacks([backup])`
# WHY-CORRECT: the runnable retries transient failures and, if the primary model keeps failing,
#              transparently fails over to a backup model — both retry AND fallback satisfied
# EXPECTED-WRONG: engine flags "no fallback/retry strategy" because resilience is composed via
#                 runnable combinators rather than an explicit try/except or loop
# CORRECT-VERDICT: no findings
"""Resilient LangChain chain with automatic retries and a backup-model fallback."""
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

_primary = ChatOpenAI(model="gpt-4.1-mini", timeout=20).with_retry(
    stop_after_attempt=4,
    wait_exponential_jitter=True,
)
_backup = ChatAnthropic(model="claude-haiku-4-5", timeout=20)

# If the retried primary still fails, calls fall over to the backup model.
resilient_llm = _primary.with_fallbacks([_backup])


async def ask(question: str) -> str:
    msg = await resilient_llm.ainvoke(question)
    return msg.content
