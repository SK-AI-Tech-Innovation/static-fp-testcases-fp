# ACE-FP-EXPECT: clean
# CATEGORY: 22_old_model_names
# SOURCE: gpt-3.5-turbo-instruct + openai-python (legacy completions endpoint)
# WHY-CORRECT: the legacy text-completion endpoint is still served by gpt-3.5-turbo-instruct (the supported successor to text-davinci-003);
#              the completions.create call shape is correct for a prompt-completion workload.
# EXPECTED-WRONG: engine flags completions-style usage / "old davinci model" as deprecated, but this is the current, valid completions model.
# CORRECT-VERDICT: no findings
"""Prompt-completion text generation via the legacy completions endpoint."""

from openai import OpenAI

client = OpenAI()

# text-davinci-003 was retired; gpt-3.5-turbo-instruct is the supported
# completions-endpoint model for code that still relies on prompt/completion
# semantics rather than the chat format.
COMPLETIONS_MODEL = "gpt-3.5-turbo-instruct"


def complete(prompt: str) -> str:
    """Complete a raw text prompt using the completions endpoint."""
    response = client.completions.create(
        model=COMPLETIONS_MODEL,
        prompt=prompt,
        max_tokens=64,
        temperature=0.7,
    )
    return response.choices[0].text.strip()


if __name__ == "__main__":
    print(complete("Write a haiku about the ocean:\n"))
