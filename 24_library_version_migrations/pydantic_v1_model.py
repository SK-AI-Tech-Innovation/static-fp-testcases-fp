# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: Pydantic v1.x model used as an LLM output schema
# WHY-CORRECT: On Pydantic v1, the public API is .dict(), .parse_obj(), .json(), and an
#   inner `class Config`. These are the correct, idiomatic methods for the v1 major version.
#   Projects pinned to pydantic<2 (common for older langchain stacks) must use this style.
# EXPECTED-WRONG: engine may flag .dict()/.parse_obj()/class Config as deprecated and demand
#   the v2 equivalents (.model_dump(), .model_validate(), model_config = ConfigDict(...)).
# CORRECT-VERDICT: no findings
"""Pydantic v1 LLM output schema using .dict(), .parse_obj(), and class Config."""
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    name: str = Field(..., description="Entity name extracted from the text")
    category: str = Field(..., description="Entity category")
    confidence: float = Field(..., ge=0.0, le=1.0)

    class Config:
        allow_mutation = False
        anystr_strip_whitespace = True


def parse_llm_output(raw: dict) -> dict:
    entity = ExtractedEntity.parse_obj(raw)
    return entity.dict()


if __name__ == "__main__":
    print(parse_llm_output({"name": "Paris", "category": "city", "confidence": 0.97}))
