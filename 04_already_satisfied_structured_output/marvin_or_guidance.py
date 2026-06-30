# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: Marvin `marvin.cast(..., target=Model)` / `@marvin.fn` typed extraction
# WHY-CORRECT: Marvin coerces model output into the declared Python/Pydantic return type and validates it;
#              `marvin.cast(text, target=Model)` returns a validated instance and `@marvin.fn` infers the
#              schema from the function's return annotation. The typed contract is enforced — no manual parsing.
# EXPECTED-WRONG: engine doesn't recognize Marvin's `cast`/`@marvin.fn` as a structured-output mechanism
#                 (no `response_format=`/`.parse(...)` token) and flags it as "free-text / not structured".
# CORRECT-VERDICT: no findings
"""Extract a typed contact and classify intent using Marvin's typed-output helpers."""
from __future__ import annotations

import marvin
from pydantic import BaseModel, Field


class Contact(BaseModel):
    name: str = Field(description="Person's full name")
    email: str = Field(description="Email address")
    company: str | None = Field(default=None, description="Company, if mentioned")


def extract_contact(text: str) -> Contact:
    # marvin.cast validates the model output against the Contact schema and returns an instance.
    return marvin.cast(text, target=Contact)


@marvin.fn
def classify_intent(message: str) -> str:
    """Classify the message intent as one of: question, complaint, request, other."""
    # Marvin enforces the str return annotation; the body is a spec, not runtime logic.
