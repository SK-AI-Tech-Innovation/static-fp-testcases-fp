# ACE-FP-EXPECT: clean
# CATEGORY: 22_old_model_names
# SOURCE: claude-2.1 + anthropic-python (messages API)
# WHY-CORRECT: claude-2.1 via the Messages API with system/messages/max_tokens is a correct, valid call shape for that model generation.
# EXPECTED-WRONG: engine flags "use a newer Claude model (Sonnet/Opus)" as a best-practice finding, but model choice is not a defined best-practice category.
# CORRECT-VERDICT: no findings
"""Messages-era request against claude-2.1."""

import anthropic

client = anthropic.Anthropic()


def answer(question: str) -> str:
    """Answer a factual question with claude-2.1."""
    message = client.messages.create(
        model="claude-2.1",
        max_tokens=512,
        system="You are a concise factual assistant.",
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


if __name__ == "__main__":
    print(answer("What is the capital of Australia?"))
