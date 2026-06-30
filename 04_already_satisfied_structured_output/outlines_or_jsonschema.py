# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: `outlines` JSON-schema-constrained generation against a Pydantic model
# WHY-CORRECT: outlines constrains decoding to the Pydantic JSON schema, so the model can only
#              emit tokens that form a valid instance; the generator returns a typed object,
#              making post-hoc parsing structurally impossible
# EXPECTED-WRONG: engine flags "missing structured output" because constrained decoding is set
#                 up via `outlines.Generator(model, Schema)` rather than a `response_format` arg
# CORRECT-VERDICT: no findings
"""Generate a typed character sheet with grammar-constrained `outlines` decoding."""
import outlines
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


class Character(BaseModel):
    name: str = Field(max_length=24)
    age: int = Field(ge=1, le=999)
    strength: int = Field(ge=1, le=20)
    weapon: str


model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
# Generator is bound to the Character schema; output is constrained to valid JSON.
generate_character = outlines.Generator(model, Character)


def make_character(prompt: str) -> Character:
    # The decoder cannot produce invalid JSON, so this is already a validated Character.
    return generate_character(prompt)
