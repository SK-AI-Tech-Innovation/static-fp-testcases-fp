# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: Pydantic v2.x model used as an LLM output schema
# WHY-CORRECT: On Pydantic v2, the idiomatic API is .model_dump(), .model_validate(), and
#   model_config = ConfigDict(...). These replace the v1 .dict()/.parse_obj()/class Config.
#   This is correct for any project on pydantic>=2 (e.g. modern langchain / openai SDK stacks).
# EXPECTED-WRONG: engine may apply v1 idioms in reverse and flag .model_dump()/.model_validate()
#   as "unknown method", suggest .dict()/.parse_obj(), or call ConfigDict an error.
# CORRECT-VERDICT: no findings
"""Pydantic v2 LLM output schema using .model_dump(), .model_validate(), and ConfigDict."""
from pydantic import BaseModel, ConfigDict, Field


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(..., description="Entity name extracted from the text")
    category: str = Field(..., description="Entity category")
    confidence: float = Field(..., ge=0.0, le=1.0)


def parse_llm_output(raw: dict) -> dict:
    entity = ExtractedEntity.model_validate(raw)
    return entity.model_dump()


if __name__ == "__main__":
    print(parse_llm_output({"name": "Paris", "category": "city", "confidence": 0.97}))
