# ACE-FP-EXPECT: clean
# CATEGORY: 27_provider_abstraction
# SOURCE: litellm.completion
# WHY-CORRECT: litellm.completion is a provider-agnostic unified call; the model string "anthropic/claude-sonnet-4-5" is the canonical LiteLLM routing format and messages follow the OpenAI chat schema.
# EXPECTED-WRONG: engine may flag the "anthropic/claude-..." model string or the non-SDK call shape as an unknown/deprecated Claude usage pattern.
# CORRECT-VERDICT: no findings
"""Provider-agnostic chat completion through LiteLLM."""

import os

import litellm


def ask(question: str) -> str:
    """Send a single user question through LiteLLM and return the reply text."""
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": question},
        ],
        max_tokens=1024,
        temperature=0.2,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(ask("Summarize the CAP theorem in one sentence."))
