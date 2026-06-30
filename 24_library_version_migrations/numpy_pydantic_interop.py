# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: numpy + Pydantic v2 — storing an embedding vector on a v2 model
# WHY-CORRECT: numpy arrays are not natively JSON-serializable, so the model stores the
#   embedding as List[float] and a validator coerces a numpy array into a plain list. This is
#   the correct v2 interop pattern: numpy stays at the compute boundary, the model holds plain
#   floats, and .tolist() bridges them. model_validate / field_validator are correct v2 API.
# EXPECTED-WRONG: engine may flag .tolist() as redundant, demand a numpy-typed field with
#   arbitrary_types_allowed, or apply pydantic v1 idioms (validator/.dict()) to this v2 model.
# CORRECT-VERDICT: no findings
"""Interop: coerce a numpy embedding into a JSON-safe List[float] on a pydantic v2 model."""
from typing import List

import numpy as np
from pydantic import BaseModel, field_validator


class EmbeddedDocument(BaseModel):
    text: str
    embedding: List[float]

    @field_validator("embedding", mode="before")
    @classmethod
    def _coerce_ndarray(cls, value):
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        return value


def embed(text: str) -> EmbeddedDocument:
    # Stand-in for a real embedding model output (a numpy float32 vector).
    vector = np.random.default_rng(0).standard_normal(8).astype(np.float32)
    return EmbeddedDocument(text=text, embedding=vector)


if __name__ == "__main__":
    doc = embed("hello world")
    print(len(doc.embedding), doc.model_dump()["embedding"][0])
