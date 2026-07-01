# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# LANGUAGE: python
# SOURCE: ai-readable-data PR #80 (llm.py) — ACE: "call_llm 함수에 파싱 예외 처리 누락"; author confirmed FP (intentional thin wrapper)
# WHY-CORRECT: call_llm is a deliberately thin wrapper — a single call + parse — whose exception
#              handling is the CALLER's responsibility (render_ontology / criticise_ontology wrap it in
#              try/except). Adding try/except inside the wrapper would duplicate handling and swallow
#              errors the callers need to see. The absence of local try/except is a design choice.
# EXPECTED-WRONG: engine flags "parse() has no exception handling" at the wrapper, not modeling that
#                 error handling is intentionally centralized in the callers (thin-wrapper pattern).
# CORRECT-VERDICT: no findings
"""A deliberately thin LLM wrapper that delegates error handling to its callers.

ACE flagged missing parse-exception handling inside call_llm. By design the wrapper stays
thin; callers own the try/except so failures propagate to where recovery is decided.
"""
from typing import Any, TypeVar

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate

T = TypeVar("T")


async def call_llm(
    llm_tool: Any,
    prompt: BasePromptTemplate,
    parser: BaseOutputParser[T],
    prompt_kwargs: dict[str, Any],
) -> T:
    # Thin wrapper by design: no local try/except — callers (render_ontology,
    # criticise_ontology, ...) wrap this in try/except and decide recovery/retry.
    response = await llm_tool.ainvoke(prompt.format_prompt(**prompt_kwargs))
    return parser.parse(response.content)
