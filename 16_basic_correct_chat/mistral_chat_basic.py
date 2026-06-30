# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: Mistral Python SDK (`mistralai`) `client.chat.complete`
# WHY-CORRECT: current mistralai 1.x shape — Mistral() client, chat.complete with model +
#              messages, reply read from choices[0].message.content. Complete and idiomatic.
# EXPECTED-WRONG: engine flags deprecated "MistralClient" usage (this uses the new Mistral class)
# CORRECT-VERDICT: no findings
"""Ask a Mistral model a single question."""
import os

from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def ask(question: str) -> str:
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask("What is the speed of light?"))
