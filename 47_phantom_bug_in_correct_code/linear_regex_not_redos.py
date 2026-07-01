# ACE-FP-EXPECT: clean
# CATEGORY: 47_phantom_bug_in_correct_code
# LANGUAGE: python
# SOURCE: ai-readable-data PR #974 (tools.py) — ACE: "_NUM_RE ReDoS(과도한 백트래킹) 위험"; author confirmed FP
# WHY-CORRECT: the pattern [0-9][0-9,]*(?:\.[0-9]+)? has no nested/overlapping quantifiers and no
#              ambiguous alternation, so each input position is consumed at most once — matching is
#              linear in input length. There is no catastrophic-backtracking surface (ReDoS).
# EXPECTED-WRONG: engine pattern-matches "regex + user input" and warns about ReDoS / catastrophic
#                 backtracking without analyzing that the quantifiers are non-nested and linear.
# CORRECT-VERDICT: no findings
"""Extract a leading number with a strictly linear regex.

ACE flagged ReDoS on _NUM_RE, but the pattern has no nested quantifiers — it cannot
backtrack catastrophically and runs in linear time.
"""
import re

# Linear: a leading digit, then digits/commas, then an optional decimal part.
# No quantifier is nested inside another, so no exponential backtracking is possible.
_NUM_RE = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?")


def parse_leading_number(text: str) -> str | None:
    m = _NUM_RE.match(text.strip())
    return m.group(0) if m else None
