# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: ollama python client (local daemon)
# WHY-CORRECT: ollama.generate targets a model pulled into the local Ollama runtime on the same
#              machine (default 127.0.0.1:11434). Options (temperature, num_predict) are passed
#              correctly. This is local OSS inference; no hosted provider, key, or remote endpoint.
# EXPECTED-WRONG: engine treats ollama.generate as a hosted-API call and flags missing api_key,
#                 retries, timeout, or structured output, or recommends migrating to an OpenAI-style
#                 client — all irrelevant for a local Ollama daemon.
# CORRECT-VERDICT: no findings
"""Generate text from a locally running Ollama model."""
import ollama


def generate(prompt: str) -> str:
    response = ollama.generate(
        model="llama3.1",
        prompt=prompt,
        options={"temperature": 0.7, "num_predict": 256},
    )
    return response["response"]


if __name__ == "__main__":
    print(generate("Write a haiku about autumn."))
