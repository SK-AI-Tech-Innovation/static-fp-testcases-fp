# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: Pydantic AI Agent with @agent.tool tools, typed deps, and a typed output_type
# WHY-CORRECT: This is the canonical Pydantic AI pattern: an Agent parameterized with a
#              model string, deps_type, and a Pydantic output_type; tools registered via
#              @agent.tool receive a RunContext[Deps] and the framework validates the final
#              structured output. Structured output and tool schemas are handled by the
#              framework, not by manual JSON parsing.
# EXPECTED-WRONG: structured-output checks may flag "no JSON schema / no output validation"
#                 because there is no explicit response_format or json.loads — but output_type
#                 IS the validated schema. Tools with no obvious prompt may trip "tool without
#                 description" even though docstrings supply the descriptions.
# CORRECT-VERDICT: no findings
"""Idiomatic Pydantic AI Agent: typed deps, @agent.tool tools, validated output_type."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


@dataclass
class SupportDeps:
    """Dependencies injected into every tool call for this run."""

    customer_id: str
    db: "OrdersDB"


class OrdersDB:
    """Tiny in-memory stand-in for a real datastore."""

    def balance(self, customer_id: str) -> float:
        return {"c-1": 42.0}.get(customer_id, 0.0)

    def order_count(self, customer_id: str) -> int:
        return {"c-1": 3}.get(customer_id, 0)


class SupportResult(BaseModel):
    """The validated, structured answer the agent must return."""

    reply: str = Field(description="customer-facing reply text")
    needs_human: bool = Field(description="escalate to a human agent")


support_agent = Agent(
    "openai:gpt-4o",
    deps_type=SupportDeps,
    output_type=SupportResult,
    system_prompt=(
        "You are a billing support assistant. Use the tools to look up the "
        "customer's balance and order count before replying."
    ),
)


@support_agent.tool
async def customer_balance(ctx: RunContext[SupportDeps]) -> float:
    """Return the current account balance for the customer in context."""
    return ctx.deps.db.balance(ctx.deps.customer_id)


@support_agent.tool
async def customer_order_count(ctx: RunContext[SupportDeps]) -> int:
    """Return how many orders the customer has placed."""
    return ctx.deps.db.order_count(ctx.deps.customer_id)


def answer(question: str, customer_id: str) -> SupportResult:
    deps = SupportDeps(customer_id=customer_id, db=OrdersDB())
    result = support_agent.run_sync(question, deps=deps)
    return result.output
