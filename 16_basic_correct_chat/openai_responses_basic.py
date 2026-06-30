# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: OpenAI Python SDK (`openai`) `client.responses.create`
# WHY-CORRECT: idiomatic Responses API call — input string passed, output read via the
#              SDK convenience property .output_text. Current, complete, correct.
# EXPECTED-WRONG: engine flags "not parsing output blocks manually" or "add streaming"
# CORRECT-VERDICT: no findings
"""Generate a short answer with the OpenAI Responses API."""
from openai import OpenAI

client = OpenAI()


def summarize(text: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"Summarize this in one sentence:\n\n{text}",
    )
    return response.output_text


if __name__ == "__main__":
    print(summarize("The quick brown fox jumps over the lazy dog."))
