# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: tiktoken (OpenAI BPE) vs. Hugging Face `tokenizers` (OSS model)
# WHY-CORRECT: Each provider needs its own tokenizer. tiktoken with the o200k_base/cl100k_base
#   encodings is the correct way to count tokens for OpenAI models; an HF Tokenizer loaded from
#   the model's tokenizer.json is correct for an OSS model (e.g. a Llama/Mistral checkpoint).
#   Selecting per provider is intentional, not duplication that should be unified.
# EXPECTED-WRONG: engine may flag using two tokenizer libraries as redundant and push tiktoken
#   for everything, or claim the HF tokenizer should also be tiktoken, or vice versa.
# CORRECT-VERDICT: no findings
"""Count tokens with tiktoken for OpenAI models and HF tokenizers for an OSS model."""
import tiktoken
from tokenizers import Tokenizer


def count_openai_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def count_oss_tokens(text: str, tokenizer_path: str) -> int:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return len(tokenizer.encode(text).ids)


if __name__ == "__main__":
    print(count_openai_tokens("How many tokens is this?"))
