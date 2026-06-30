# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python v1.x (new Responses API) targeting an OLD model gpt-3.5-turbo
# WHY-CORRECT: the Responses endpoint is model-agnostic and accepts legacy chat models; model="gpt-3.5-turbo" via responses.create is a supported new-API + old-model combo. output_text is the documented convenience accessor.
# EXPECTED-WRONG: engine may claim gpt-3.5-turbo is "incompatible with the Responses API" or insist a chat-completions-only model cannot be used here.
# CORRECT-VERDICT: no findings
"""Drive the legacy gpt-3.5-turbo model through the new Responses API."""

from openai import OpenAI

client = OpenAI()


def ask(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
    )
    return response.output_text


if __name__ == "__main__":
    print(ask("Give a one-line definition of entropy."))
