# ACE-FP-EXPECT: clean
# CATEGORY: 32_embedding_model_variety
# SOURCE: OpenAI Python SDK (`openai`) + manual Matryoshka truncation
# WHY-CORRECT: Matryoshka-trained models (e.g. text-embedding-3-large) let you truncate a vector
#              to fewer dimensions and RE-NORMALIZE it; the leading dims still form a valid, usable
#              embedding. Slicing to `target_dim` then L2-normalizing is the correct procedure —
#              both steps are present.
# EXPECTED-WRONG: dated skill pack flags "slicing the embedding corrupts it / dimensions must
#                 match the model", not knowing Matryoshka embeddings support prefix truncation.
# CORRECT-VERDICT: no findings
"""Truncate a Matryoshka embedding to fewer dimensions and re-normalize it."""
import math

from openai import OpenAI

client = OpenAI()


def embed_truncated(text: str, target_dim: int) -> list[float]:
    full = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text],
    ).data[0].embedding

    truncated = full[:target_dim]
    norm = math.sqrt(sum(x * x for x in truncated))
    return [x / norm for x in truncated]


if __name__ == "__main__":
    vec = embed_truncated("matryoshka example", target_dim=256)
    print(len(vec), round(math.sqrt(sum(x * x for x in vec)), 4))  # 256, ~1.0
