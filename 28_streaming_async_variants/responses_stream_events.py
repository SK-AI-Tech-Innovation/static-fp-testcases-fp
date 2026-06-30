# ACE-FP-EXPECT: clean
# CATEGORY: 28_streaming_async_variants
# SOURCE: openai-python v1.x (Responses API, event-based streaming)
# WHY-CORRECT: client.responses.stream(...) yields typed semantic events; matching on event.type (response.output_text.delta) and reading event.delta, then get_final_response(), is the documented Responses streaming pattern.
# EXPECTED-WRONG: engine may claim Responses API has no .stream(), or demand choices[0].delta (Chat shape) instead of event.delta.
# CORRECT-VERDICT: no findings
"""Event-based streaming using the OpenAI Responses API."""

from openai import OpenAI

client = OpenAI()


def stream_response(prompt: str) -> str:
    text_parts = []
    with client.responses.stream(
        model="gpt-4o",
        input=prompt,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                text_parts.append(event.delta)
                print(event.delta, end="", flush=True)
            elif event.type == "response.error":
                raise RuntimeError(event.error)

        final = stream.get_final_response()

    print()
    print("status:", final.status)
    return "".join(text_parts)


if __name__ == "__main__":
    stream_response("Write a haiku about streaming data.")
