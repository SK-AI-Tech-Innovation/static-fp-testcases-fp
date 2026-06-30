# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: A 3D math/physics Vector class (dot, cross, normalize) for geometry
# WHY-CORRECT: "vector" here is a Euclidean 3D vector with dot/cross products and a norm.
#              These are geometry operations; there are no embeddings, no embedding model,
#              no vector database, and no similarity search over learned representations.
# EXPECTED-WRONG: keyword "vector" / "embed"(in "embedding"? no) / cosine-like dot product
#                 -> false "embedding / vector store" detection -> spurious "no embedding
#                 model configured" finding.
# CORRECT-VERDICT: no findings
"""Immutable 3D math/physics vector with dot, cross, and normalize. Not embeddings."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector3:
    """A 3D Euclidean vector used for geometry and physics calculations."""

    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> "Vector3":
        """Return this vector multiplied by a scalar factor."""
        return Vector3(self.x * factor, self.y * factor, self.z * factor)

    def dot(self, other: "Vector3") -> float:
        """Scalar (dot) product of two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        """Vector (cross) product, yielding a perpendicular vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        """Euclidean length (L2 norm) of the vector."""
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vector3":
        """Return a unit vector pointing in the same direction."""
        length = self.magnitude()
        if length == 0.0:
            raise ValueError("cannot normalize a zero vector")
        return self.scale(1.0 / length)


def angle_between(a: Vector3, b: Vector3) -> float:
    """Angle in radians between two vectors via the dot product."""
    denom = a.magnitude() * b.magnitude()
    if denom == 0.0:
        raise ValueError("angle undefined for a zero vector")
    cosine = max(-1.0, min(1.0, a.dot(b) / denom))
    return math.acos(cosine)
