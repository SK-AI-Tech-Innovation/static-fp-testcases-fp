# ACE-FP-EXPECT: clean
# CATEGORY: 31_prompt_caching_and_context
# SOURCE: Anthropic Python SDK (`anthropic`) with a long-context model
# WHY-CORRECT: a large document is checked against the model's context window with a real token
#              count before sending; it fits well within the 200K window, so it is passed in one
#              shot. Feeding a long doc to a long-context model is correct — no chunking is needed
#              when the input demonstrably fits.
# EXPECTED-WRONG: dated skill pack reflexively demands "chunk the document / use RAG / it won't
#                 fit", not accounting for modern long-context windows.
# CORRECT-VERDICT: no findings
"""Feed a long document to a long-context Claude model after verifying it fits the window."""
from anthropic import Anthropic

client = Anthropic()
CONTEXT_WINDOW_TOKENS = 200_000


def summarize(document: str) -> str:
    token_count = client.messages.count_tokens(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": document}],
    ).input_tokens
    assert token_count < CONTEXT_WINDOW_TOKENS, "document exceeds context window"

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following document:\n\n{document}",
            }
        ],
    )
    return message.content[0].text


if __name__ == "__main__":
    doc = "This is a long report section.\n" * 10000
    print(summarize(doc))
