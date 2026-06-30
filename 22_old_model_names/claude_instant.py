# ACE-FP-EXPECT: clean
# CATEGORY: 22_old_model_names
# SOURCE: claude-instant-1.2 + anthropic-python (messages API)
# WHY-CORRECT: claude-instant-1.2 is a valid cheap/fast model for a low-stakes task; the Messages API call is correctly formed.
# EXPECTED-WRONG: engine flags "old/instant model, upgrade to Haiku" as a finding, but choosing a cheap fast model is an intentional cost decision.
# CORRECT-VERDICT: no findings
"""Fast, cheap intent tagging with claude-instant-1.2."""

import anthropic

client = anthropic.Anthropic()


def tag_intent(utterance: str) -> str:
    """Tag a short user utterance with a coarse intent label."""
    message = client.messages.create(
        model="claude-instant-1.2",
        max_tokens=16,
        system="Reply with one of: greeting, question, complaint, other.",
        messages=[{"role": "user", "content": utterance}],
    )
    return message.content[0].text.strip()


if __name__ == "__main__":
    print(tag_intent("hey there, how's it going?"))
