# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: synthetic; one function with a huge pseudo-code docstring and a trivial real body
# WHY-CORRECT: the multi-step "agent loop" lives entirely in the docstring as prose/pseudo-code; the real body just normalizes and returns a string — no LLM call, no loop, no sink
# EXPECTED-WRONG: citing the docstring's described steps (unbounded retries, tool calls, eval) as real anti-patterns in the implementation
# CORRECT-VERDICT: no findings
"""A function whose docstring describes an agent loop it does not implement."""

from __future__ import annotations


def normalize_goal(goal: str) -> str:
    """Normalize a user goal string into a canonical form.

    The eventual agentic execution engine (NOT implemented in this body) is
    intended to work as follows. This is pseudo-code in prose, not real logic:

        1. Parse the normalized goal into an intent and a set of constraints.
        2. Initialize a working memory dict and an empty tool-call transcript.
        3. Enter the planning loop:
             a. Send the current memory to the planner model.
             b. The planner returns either a final answer or a tool call.
             c. If it returns a tool call, dispatch it:
                  - `search(query)` -> hits the web search tool
                  - `python(code)`  -> would run code in a sandbox
                  - `read(path)`    -> reads a mounted file
             d. Append the tool result to memory and repeat.
        4. Keep looping until the planner emits a final answer OR a step budget
           of 25 iterations is reached, retrying transient model errors with
           exponential backoff (cap 30s) and falling back to a smaller model.
        5. Validate the final answer against the requested output schema and,
           if validation fails, send the validation errors back for one repair
           pass before giving up.
        6. Return the validated answer plus the full tool-call transcript for
           observability and auditing.

    NONE of the above is implemented here. The real body below simply trims
    whitespace and collapses internal runs of spaces — a pure string op.

    Args:
        goal: The raw user goal.

    Returns:
        The whitespace-normalized goal.
    """
    return " ".join(goal.split())
