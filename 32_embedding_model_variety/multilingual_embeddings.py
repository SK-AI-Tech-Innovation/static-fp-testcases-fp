# ACE-FP-EXPECT: clean
# CATEGORY: 32_embedding_model_variety
# SOURCE: sentence-transformers with BAAI/bge-m3 (multilingual)
# WHY-CORRECT: bge-m3 is a multilingual model that embeds many languages into one shared space,
#              so mixed-language inputs can be encoded together in a single call. Normalizing the
#              output for cosine similarity is correct. The usage is complete for this model.
# EXPECTED-WRONG: dated skill pack flags "you mix languages — detect language / use per-language
#                 models first", missing that bge-m3 is explicitly multilingual by design.
# CORRECT-VERDICT: no findings
"""Embed mixed-language text with the multilingual bge-m3 model."""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")


def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()


if __name__ == "__main__":
    vectors = embed(["Hello world", "안녕하세요 세계", "Bonjour le monde"])
    print(len(vectors), len(vectors[0]))
