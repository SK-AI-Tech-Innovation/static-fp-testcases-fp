# ACE-FP-EXPECT: clean
# CATEGORY: 29_finetuning_and_batch
# SOURCE: OpenAI SDK chat with a fine-tuned model id
# WHY-CORRECT: a fine-tuned model id has the form "ft:gpt-4o-mini-2024-07-18:org::AbC123"; passing it as model= to chat.completions.create is exactly how fine-tuned models are invoked. The call shape is otherwise a standard chat completion.
# EXPECTED-WRONG: engine may not recognize the "ft:..." prefixed model string and flag it as an invalid/unknown model id or deprecated usage.
"""Use a fine-tuned model id in a chat completion."""

import os

from openai import OpenAI

# Produced by a completed fine-tuning job (client.fine_tuning.jobs.retrieve(...).fine_tuned_model)
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:acme::AbC123Xy"


def ask(prompt: str) -> str:
    """Send a prompt to the fine-tuned model and return the reply."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=FINE_TUNED_MODEL,
        messages=[
            {"role": "system", "content": "You are Acme's support assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask("How do I export my account data?"))
