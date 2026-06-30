# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: Anthropic Python SDK + Pydantic (typed/validated tool result)
# WHY-CORRECT: The tool produces a Pydantic model, validates it, and serializes
#   it deterministically (model_dump_json) before feeding it back as the
#   tool_result content keyed by tool_use_id. The result is typed end to end and
#   fed back to the model in the documented shape — nothing is missing.
# EXPECTED-WRONG: engine may suggest "validate the tool output" or "feed the
#   result back to the model" — both already done with a typed, validated model.
# CORRECT-VERDICT: no findings
"""A tool that returns a typed/validated result, fed back correctly."""

import anthropic
from pydantic import BaseModel, Field

client = anthropic.Anthropic()


class StockQuote(BaseModel):
    """Validated structured result returned by the quote tool."""

    symbol: str = Field(description="Ticker symbol, uppercase.")
    price_usd: float = Field(description="Latest trade price in USD.", gt=0)
    currency: str = Field(default="USD", description="ISO currency code.")


TOOLS = [
    {
        "name": "get_stock_quote",
        "description": "Get the latest stock quote for a ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."},
            },
            "required": ["symbol"],
        },
    }
]


def get_stock_quote(symbol: str) -> StockQuote:
    """Produce a validated, typed quote (stubbed deterministic data)."""
    return StockQuote(symbol=symbol.upper(), price_usd=187.42)


def quote(question: str) -> str:
    """Run one tool turn; the typed result is serialized back to the model."""
    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        tools=TOOLS,
        messages=messages,
    )

    if response.stop_reason != "tool_use":
        return next((b.text for b in response.content if b.type == "text"), "")

    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = get_stock_quote(block.input["symbol"])
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.model_dump_json(),
                }
            )
    messages.append({"role": "user", "content": tool_results})

    final = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        tools=TOOLS,
        messages=messages,
    )
    return next((b.text for b in final.content if b.type == "text"), "")


if __name__ == "__main__":
    print(quote("What's the latest price of AAPL?"))
