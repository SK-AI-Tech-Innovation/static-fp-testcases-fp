# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: Pydantic v2 installed, but using the `pydantic.v1` compatibility namespace
# WHY-CORRECT: Pydantic v2 ships a built-in `pydantic.v1` namespace that exposes the full v1
#   API. Libraries that still emit v1 models (older langchain core, some SDKs) require v1
#   BaseModel; importing from pydantic.v1 lets v1-style code run unchanged on a v2 install.
#   Using .dict()/.parse_obj() here is correct BECAUSE these are v1 objects.
# EXPECTED-WRONG: engine may flag the `pydantic.v1` import as wrong / "use top-level pydantic",
#   or flag .dict()/.parse_obj() as deprecated v1 calls without recognizing the compat namespace.
# CORRECT-VERDICT: no findings
"""Deliberate use of the pydantic.v1 compat namespace for a v1-only LLM tool schema."""
from pydantic.v1 import BaseModel, Field


class ToolArgs(BaseModel):
    """v1 model required by a dependency that has not migrated to pydantic v2."""

    query: str = Field(..., description="Search query string")
    top_k: int = Field(5, ge=1, le=50)


def normalize_args(raw: dict) -> dict:
    args = ToolArgs.parse_obj(raw)
    return args.dict()


if __name__ == "__main__":
    print(normalize_args({"query": "vector databases", "top_k": 3}))
