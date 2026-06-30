# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: langsmith + langchain-openai
# WHY-CORRECT: With LANGCHAIN_TRACING_V2 enabled and a LangChainTracer callback attached, every model invocation is logged to LangSmith as a run with inputs, outputs, latency, and token usage. The @traceable wrapper extends tracing to the orchestrating function. Observability is fully handled.
# EXPECTED-WRONG: missing observability / no tracing or token tracking around the LLM call
# CORRECT-VERDICT: no findings
"""Trace LLM runs to LangSmith via env config and a LangChainTracer callback."""

import os

from langchain_core.messages import HumanMessage
from langchain_core.tracers import LangChainTracer
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Enable tracing; the SDK then ships run trees to LangSmith automatically.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "chat-service")

tracer = LangChainTracer()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[tracer])


@traceable(run_type="chain", name="answer_question")
def answer(question: str) -> str:
    """Captured as a LangSmith run tree with token usage and latency recorded."""
    result = llm.invoke([HumanMessage(content=question)])
    return result.content
