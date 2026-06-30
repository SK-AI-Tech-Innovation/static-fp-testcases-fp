# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: OpenAI Responses API `responses.parse(text_format=Model)` with a NESTED Pydantic schema
# WHY-CORRECT: `text_format` accepts arbitrarily nested Pydantic models (sub-models, lists of sub-models);
#              the SDK builds the recursive JSON schema and returns a fully validated `output_parsed` object
#              graph. Nested structure is enforced end-to-end — the consumer never parses raw JSON.
# EXPECTED-WRONG: engine only recognizes flat `response_format=Pydantic` from the dated example and either
#                 flags the nested model as "not structured" or claims nested schemas are unsupported and
#                 proposes flattening / an anachronistic chat-completions downgrade.
# CORRECT-VERDICT: no findings
"""Extract a nested order (customer + line items) using the OpenAI Responses API."""
from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()


class Customer(BaseModel):
    name: str = Field(description="Customer's full name")
    email: str = Field(description="Customer email address")


class LineItem(BaseModel):
    sku: str = Field(description="Product SKU")
    quantity: int = Field(description="Number of units ordered")
    unit_price: float = Field(description="Price per unit")


class Order(BaseModel):
    order_id: str = Field(description="Unique order identifier")
    customer: Customer
    items: list[LineItem] = Field(default_factory=list)


def parse_order(text: str) -> Order:
    response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Extract the order into the nested schema."},
            {"role": "user", "content": text},
        ],
        text_format=Order,
    )
    # output_parsed is a fully validated Order with a Customer and a list of LineItems.
    return response.output_parsed
