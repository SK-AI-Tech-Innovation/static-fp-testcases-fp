# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: OpenAI Python SDK Responses API streaming (`client.responses.stream`)
# WHY-CORRECT: streaming text deltas is the intended use of `responses.stream`; structured-output-per-token
#              is not applicable to a free-form streamed answer. Event handling here is idiomatic and complete.
# EXPECTED-WRONG: engine flags "missing structured output / response_format not set" on a streaming call where
#                 a typed schema is intentionally not used, or mis-detects the `with client...stream() as s`
#                 context-manager + event loop as an unhandled/raw completion.
# CORRECT-VERDICT: no findings
"""Stream an assistant answer token-by-token with the OpenAI Responses streaming API."""
from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

client = OpenAI()


def stream_answer(question: str) -> Iterator[str]:
    """Yield text deltas as they arrive from the model."""
    with client.responses.stream(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": question}],
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
        # Ensure the stream is fully drained and any terminal error surfaces.
        stream.get_final_response()


if __name__ == "__main__":
    for chunk in stream_answer("Explain backpressure in one paragraph."):
        print(chunk, end="", flush=True)
