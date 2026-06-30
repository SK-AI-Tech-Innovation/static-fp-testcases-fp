# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: huggingface_hub InferenceClient against a self-hosted TGI endpoint
# WHY-CORRECT: InferenceClient points at a locally running Text Generation Inference server
#              (localhost:8080) serving an open-weights model. text_generation is called with
#              explicit decoding params. This is self-hosted OSS serving, not a hosted commercial API.
# EXPECTED-WRONG: engine sees an InferenceClient + a URL and assumes a hosted provider, then flags a
#                 missing API key / token, retries, or structured output, or treats text_generation
#                 as the wrong (chat-completions) shape for a local TGI server.
# CORRECT-VERDICT: no findings
"""Call a self-hosted TGI server via huggingface_hub InferenceClient."""
from huggingface_hub import InferenceClient

client = InferenceClient(base_url="http://localhost:8080")


def generate(prompt: str) -> str:
    return client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )


if __name__ == "__main__":
    print(generate("Explain what a vector database is in plain terms:"))
