# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: Guidance constrained generation: @guidance function with gen/select constraints
# WHY-CORRECT: This is idiomatic Guidance: a @guidance-decorated function appends literal
#              text and constrained generation primitives (gen with a stop/regex, select
#              over a fixed option set) to the model `lm`, then reads results back via
#              lm["name"]. The constraints guarantee well-formed output; the model object
#              is invoked through the Guidance grammar, not a raw client call.
# EXPECTED-WRONG: the lm += "..." accumulation and lm["field"] access look unusual and may
#                 be flagged as string concatenation of prompts / unvalidated output, even
#                 though select() and the gen regex ARE the structured-output constraints.
# CORRECT-VERDICT: no findings
"""Idiomatic Guidance: constrained program using gen() and select() over a model."""
from __future__ import annotations

import guidance
from guidance import gen, models, select


@guidance
def classify_ticket(lm, ticket: str):
    """Constrain the model to a sentiment label, a priority, and a short reason."""
    lm += f"Support ticket:\n{ticket}\n\n"
    lm += "Sentiment: " + select(["positive", "neutral", "negative"], name="sentiment")
    lm += "\nPriority: " + select(["low", "medium", "high"], name="priority")
    lm += "\nReason: " + gen("reason", max_tokens=40, stop="\n")
    return lm


def triage(ticket: str) -> dict[str, str]:
    """Run the constrained program and return the structured fields."""
    lm = models.OpenAI("gpt-4o-mini")
    lm = lm + classify_ticket(ticket)
    return {
        "sentiment": lm["sentiment"],
        "priority": lm["priority"],
        "reason": lm["reason"].strip(),
    }
