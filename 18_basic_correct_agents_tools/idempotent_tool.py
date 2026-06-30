# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: Anthropic Python SDK tool definition + idempotency-key pattern
# WHY-CORRECT: The tool is designed to be safe to retry: callers pass an
#   idempotency_key, the handler deduplicates on that key so a repeated call
#   returns the original result instead of charging twice. The schema documents
#   the key and requires it. This is the canonical idempotent-side-effect tool,
#   so retries are safe by construction — nothing to harden.
# EXPECTED-WRONG: engine may suggest "make this tool idempotent" or "guard
#   against duplicate side effects on retry" — already implemented via the key.
# CORRECT-VERDICT: no findings
"""A tool designed to be idempotent (safe retries) via an idempotency key."""

CHARGE_TOOL = {
    "name": "charge_card",
    "description": (
        "Charge a customer's card. Safe to retry: pass the same idempotency_key "
        "to retry a charge without double-charging — a repeated key returns the "
        "original result."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "The customer to charge.",
            },
            "amount_cents": {
                "type": "integer",
                "description": "Amount to charge, in cents.",
                "minimum": 1,
            },
            "idempotency_key": {
                "type": "string",
                "description": (
                    "Unique key for this logical charge. Reusing the same key "
                    "returns the original result instead of charging again."
                ),
            },
        },
        "required": ["customer_id", "amount_cents", "idempotency_key"],
    },
}

# In production this would be a durable store (Redis/DB); a dict suffices here.
_PROCESSED: dict[str, dict] = {}


def charge_card(customer_id: str, amount_cents: int, idempotency_key: str) -> dict:
    """Charge a card idempotently: repeats of a key return the first result."""
    if idempotency_key in _PROCESSED:
        return _PROCESSED[idempotency_key]

    # Perform the side effect exactly once per key.
    result = {
        "charge_id": f"ch_{idempotency_key}",
        "customer_id": customer_id,
        "amount_cents": amount_cents,
        "status": "succeeded",
    }
    _PROCESSED[idempotency_key] = result
    return result


if __name__ == "__main__":
    first = charge_card("cus_42", 1500, "order-9981")
    retry = charge_card("cus_42", 1500, "order-9981")  # safe retry
    assert first == retry  # same key -> same result, charged once
    print(first)
