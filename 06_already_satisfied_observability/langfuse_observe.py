# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: Langfuse `@observe()` decorator + `update_current_observation` usage capture
# WHY-CORRECT: every call is wrapped in a Langfuse trace/generation that records the model,
#              prompt/completion tokens, and latency automatically — observability satisfied
# EXPECTED-WRONG: engine flags "no logging/metrics on LLM call" because there is no explicit
#                 logger or print; the telemetry is emitted by the decorator/context
# CORRECT-VERDICT: no findings
"""Traced chat completion with Langfuse capturing model, tokens, and latency."""
from langfuse import observe
from langfuse.openai import openai  # drop-in client that auto-reports usage to Langfuse

client = openai.OpenAI()


@observe(name="answer-question")
def answer(question: str) -> str:
    # The langfuse.openai wrapper records the generation span with token usage and latency;
    # @observe links it into the surrounding trace. No manual metric plumbing needed.
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content
