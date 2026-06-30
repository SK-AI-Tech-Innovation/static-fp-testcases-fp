# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: LangChain `Runnable.with_structured_output` (langchain-core / langchain-openai)
# WHY-CORRECT: schema is enforced by with_structured_output(Model) — the parsing layer is gone, principle satisfied
# EXPECTED-WRONG: engine flags "free-text parsing / not using structured output" because the call shape
#                 differs from the skill example's `beta.chat.completions.parse`
# CORRECT-VERDICT: no findings
"""Classify a support ticket into a typed schema using LangChain structured output."""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class TicketTriage(BaseModel):
    category: str = Field(description="One of: billing, bug, feature_request, account")
    priority: str = Field(description="One of: low, medium, high, urgent")
    needs_human: bool = Field(description="True if the ticket requires a human agent")


_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
_triager = _llm.with_structured_output(TicketTriage)


async def triage_ticket(ticket_text: str) -> TicketTriage:
    # No json.loads, no regex — the runnable returns a validated TicketTriage instance.
    return await _triager.ainvoke(
        [
            ("system", "Triage the support ticket into the schema."),
            ("human", ticket_text),
        ]
    )
