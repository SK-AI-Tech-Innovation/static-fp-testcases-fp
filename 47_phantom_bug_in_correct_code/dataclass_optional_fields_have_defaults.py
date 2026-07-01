# ACE-FP-EXPECT: clean
# CATEGORY: 47_phantom_bug_in_correct_code
# LANGUAGE: python
# SOURCE: ace PR #960 — ACE: "생성자 시그니처 불일치 가능성 (추측성, conf 0.25)"; author confirmed FP
# WHY-CORRECT: the dataclass has 3 required fields (slug/description/detect) and the rest carry
#              defaults, so a call passing the required fields (and optionally some extras) is a
#              complete, valid constructor call. There is no signature mismatch.
# EXPECTED-WRONG: engine speculates ("추측성") that a constructor call might be missing arguments or
#                 leaving fields underfilled, without confirming the dataclass field defaults.
# CORRECT-VERDICT: no findings
"""A dataclass instantiation that fully satisfies the constructor.

ACE raised a low-confidence speculative "signature mismatch" on this call; the dataclass
below shows the optional fields all have defaults, so the call is complete.
"""
from dataclasses import dataclass, field


@dataclass
class DetectionPattern:
    slug: str
    description: str
    detect: str
    owasp_llm: str | None = None
    owasp_asi: str | None = None
    atlas: str | None = None
    default_severity: str = "warning"
    tags: list[str] = field(default_factory=list)


# Required fields plus several optionals supplied — a complete, valid construction.
PATTERN = DetectionPattern(
    slug="memory-write-no-guard",
    description="Unguarded write to long-term memory store",
    detect="memory.write(",
    owasp_llm="LLM06",
    default_severity="critical",
)
