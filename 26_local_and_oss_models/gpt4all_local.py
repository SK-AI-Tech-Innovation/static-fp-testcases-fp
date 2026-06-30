# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: gpt4all local model chat session
# WHY-CORRECT: GPT4All loads a quantized model file and runs generation inside a local chat_session
#              context manager. Everything is on-device; "gpt4all" is a local runtime, not the
#              OpenAI GPT-4 API despite the name. Generation params are set explicitly.
# EXPECTED-WRONG: engine sees "gpt4" / "GPT4All" / chat_session and assumes a hosted OpenAI GPT-4
#                 call, then flags a missing API key, retries, or structured output — none of which
#                 apply to a local GPT4All session.
# CORRECT-VERDICT: no findings
"""Local chat with a GPT4All quantized model."""
from gpt4all import GPT4All

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", allow_download=False)


def ask(prompt: str) -> str:
    with model.chat_session():
        return model.generate(prompt, max_tokens=300, temp=0.7)


if __name__ == "__main__":
    print(ask("Give me a fun fact about octopuses."))
