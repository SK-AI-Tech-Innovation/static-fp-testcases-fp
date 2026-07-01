# ACE-FP-EXPECT: clean
# CATEGORY: 47_phantom_bug_in_correct_code
# LANGUAGE: python
# SOURCE: web research (deep-research 2026-06-30) — arXiv 2408.13855 "Missing whitelisted cases" (largest static-analysis FP root cause); SonarQube S3077 analog; OWASP SQLi cheat sheet (allow-listed identifier may be concatenated)
# WHY-CORRECT: the LLM-chosen value is constrained to a fixed allow-list (Pydantic Literal) BEFORE it
#              reaches the sink, and the dict maps that validated key to a fixed, hardcoded argv / column.
#              The model can only select one of the enumerated safe values — there is no untrusted-input
#              path to the SQL string or the subprocess, so this is the validated mirror of an
#              unvalidated-output-to-sink anti-pattern, and must stay clean.
# EXPECTED-WRONG: engine sees "LLM output -> SQL/subprocess" and flags injection / "unvalidated output to
#                 sink", not modeling that the Literal type + allow-list lookup already confined the value
#                 to a safe, closed set (the classic "missing whitelisted case" false positive).
# CORRECT-VERDICT: no findings
"""LLM action routed through a Literal allow-list before reaching a sink.

ACE flags raw model-output-to-sink as injection (correctly, when unvalidated). Here the
value is constrained to a Pydantic Literal and mapped to a fixed command/column, so the
sink only ever sees a safe, enumerated value — the engine must NOT flag it.
"""
import subprocess
from typing import Literal

from pydantic import BaseModel


class AgentAction(BaseModel):
    # The model's choice is constrained to this closed set by schema validation.
    name: Literal["disk_usage", "list_files", "uptime"]


# Validated key -> fixed argv. No model text is interpolated into the command.
_ALLOWED_ARGV: dict[str, list[str]] = {
    "disk_usage": ["df", "-h"],
    "list_files": ["ls", "-la"],
    "uptime": ["uptime"],
}


def run_agent_action(action: AgentAction) -> str:
    # action.name is guaranteed by the Literal type to be one of the three safe keys.
    argv = _ALLOWED_ARGV[action.name]  # fixed argv, no shell, no interpolation
    result = subprocess.run(argv, capture_output=True, text=True)  # noqa: S603 — argv is a fixed allow-list value
    return result.stdout
