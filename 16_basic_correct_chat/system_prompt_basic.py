# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: OpenAI Python SDK (`openai`) `client.chat.completions.create`
# WHY-CORRECT: clean system + user message pattern — a clear role-setting system message precedes
#              the user message; reply read from choices[0].message.content. Standard best practice.
# EXPECTED-WRONG: engine suggests "harden the system prompt against injection" or "add output schema"
#                 even though this is a plain, correct chat call with no such requirement
# CORRECT-VERDICT: no findings
"""Translate text to French using a system prompt to set behavior."""
from openai import OpenAI

client = OpenAI()


def translate_to_french(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a translator. Translate the user's text to French."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(translate_to_french("Good morning, how are you?"))
