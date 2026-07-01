# ACE-FP-EXPECT: clean
# CATEGORY: 48_security_false_positive_trusted_source
# LANGUAGE: python
# SOURCE: ai-readable-data PR #333 (render_facts/prompt.py) — ACE: "Prompt injection: 외부 입력 {system_preamble_semantic} 삽입"; author confirmed FP
# WHY-CORRECT: {system_preamble_semantic} is filled from a module-level constant defined in our own
#              source (common_prompts.py), not from any external/user/request input. There is no
#              attacker-controllable vector, so it is not a prompt-injection surface.
# EXPECTED-WRONG: engine classifies a template variable as "external input" purely from the name and
#                 the .format() call, flagging prompt injection — misreading a static in-code constant
#                 as untrusted data.
# CORRECT-VERDICT: no findings
"""Compose a prompt from a module-level constant preamble (not external input).

ACE flagged prompt injection on {system_preamble_semantic}; that placeholder is filled
from an in-code constant, with no user/external path into it.
"""
# Module-level constant — part of our own source, not external input.
system_preamble_semantic = (
    "# SYSTEM INSTRUCTION\n"
    "You are an expert in semantic technologies, SPARQL and triple extraction."
)

_TEMPLATE = (
    "{system_preamble_semantic}\n\n"
    "Extract triples from the following document:\n{document}"
)


def render_prompt(document: str) -> str:
    # system_preamble_semantic is the trusted in-code constant above; only `document`
    # is data, and it is placed as content (not as instructions).
    return _TEMPLATE.format(
        system_preamble_semantic=system_preamble_semantic,
        document=document,
    )
