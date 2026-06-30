# ACE-FP-EXPECT: clean
# CATEGORY: 19_basic_correct_embeddings_and_misc
# SOURCE: OpenAI Python SDK (`openai`) `chat.completions.create(stream=True)`
# WHY-CORRECT: textbook token streaming — iterate chunks, guard the optional delta content, print
#              incrementally without buffering. A plain free-text stream needs no schema.
# EXPECTED-WRONG: engine suggests "use structured output" or "validate the response" — inappropriate
#                 for a free-form streamed answer printed to stdout.
# CORRECT-VERDICT: no findings
"""Stream a chat completion token-by-token to stdout."""
from openai import OpenAI

client = OpenAI()


def stream_answer(question: str) -> None:
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": question}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    stream_answer("Write a haiku about the sea.")
