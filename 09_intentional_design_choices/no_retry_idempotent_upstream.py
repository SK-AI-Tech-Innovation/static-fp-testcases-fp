# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A worker handler invoked from an at-least-once message queue
# WHY-CORRECT: No in-function retry on purpose; the queue/worker layer already retries idempotent jobs, and adding inner retries would cause duplicate work and double-counted backoff
# EXPECTED-WRONG: Engine flags "missing retry/error handling" on the LLM call and suggests wrapping it in a retry loop, duplicating the queue's responsibility
# CORRECT-VERDICT: no findings
"""Summarize a document inside a queue worker (retries owned by the queue)."""

from anthropic import Anthropic

client = Anthropic()


def handle_summarize_job(document_text: str) -> str:
    """Produce a summary for one queued job.

    Retry policy note:
        This function intentionally performs NO retry. It runs under an
        idempotent, at-least-once job queue: on failure the message is
        re-delivered and the worker re-invokes this handler. Adding a retry
        loop here would multiply attempts and corrupt the queue's backoff
        accounting. Errors are allowed to propagate so the queue can re-drive.

    Args:
        document_text: The full text to summarize.

    Returns:
        str: A concise summary.
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[
            {"role": "user", "content": f"Summarize:\n\n{document_text}"}
        ],
    )
    return response.content[0].text
