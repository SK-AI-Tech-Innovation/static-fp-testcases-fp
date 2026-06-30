# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: llama-cpp-python (local GGUF inference)
# WHY-CORRECT: Llama loads a quantized GGUF model from disk and runs create_chat_completion fully
#              on-device (CPU/Metal/CUDA via n_gpu_layers). The messages shape mirrors OpenAI's only
#              for ergonomics; there is no network, key, or endpoint. Params are complete.
# EXPECTED-WRONG: engine matches create_chat_completion(messages=...) to the hosted OpenAI chat
#                 pattern and emits API-shaped findings (add max_retries, set timeout, use a remote
#                 model id), none of which apply to a local llama.cpp binding.
# CORRECT-VERDICT: no findings
"""Local GGUF chat inference via llama-cpp-python."""
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama-3.1-8b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
)


def chat(user_message: str) -> str:
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful local assistant."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(chat("Summarize the water cycle in two sentences."))
