# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: synthetic; a prompt-template module whose bulk is example code inside a string literal
# WHY-CORRECT: the "code" (including an eval/exec and a no-retry call) lives inside a prompt-template STRING, not executable logic; the only real code returns a formatted string
# EXPECTED-WRONG: flagging the eval()/exec() or the bare requests call that appear inside the PROMPT_TEMPLATE string as if they were real sinks/anti-patterns
# CORRECT-VERDICT: no findings
"""A prompt-template module. Its main content is a big string literal.

The template embeds illustrative pseudo-code (including an `eval` and a call
with no retry) purely as text shown to the model. None of it executes. The
only executable function is `render_prompt`, which formats the template.
"""

# The string below is a PROMPT shown to an LLM. It contains example code as
# documentation for the model — it is data, never executed by this module.
PROMPT_TEMPLATE = '''
You are a Python tutor. A student submitted this snippet:

    def compute(user_formula, variables):
        # Example of an UNSAFE pattern the student wrote:
        return eval(user_formula, {}, variables)

    def fetch(url):
        import requests
        # Note: this call has no retry and no timeout
        return requests.get(url).json()

Explain to the student, in plain language, why evaluating arbitrary input
with eval() is dangerous, and why a network call should have a timeout and
a retry policy. Do NOT execute the code. Respond with a short explanation
and a corrected version that uses ast.literal_eval and a bounded retry loop.

Student question: {question}
'''


def render_prompt(question: str) -> str:
    """Fill the template with the student's question. This is the only logic."""
    return PROMPT_TEMPLATE.format(question=question)
