# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A chat handler that streams tokens to the user as they arrive
# WHY-CORRECT: Output is streamed plain text; structured-output/JSON parsing is intentionally not used because tokens are forwarded incrementally to the UI
# EXPECTED-WRONG: Engine suggests using structured output / response_format / parsing the result into a schema, which is impractical for streaming
# CORRECT-VERDICT: no findings
"""Stream assistant tokens to the caller as they are generated."""

from anthropic import Anthropic

client = Anthropic()


def stream_reply(user_message: str):
    """Yield text chunks of the assistant reply as they arrive.

    Args:
        user_message: The user's latest message.

    Yields:
        str: Incremental text deltas suitable for live UI rendering.
    """
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            # Forward each delta straight to the consumer; no buffering or
            # JSON parsing because the contract here is free-form streamed text.
            yield text
