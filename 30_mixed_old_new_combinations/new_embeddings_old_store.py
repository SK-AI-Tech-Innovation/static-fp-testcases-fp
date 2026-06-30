# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python v1.x embeddings (text-embedding-3-large with dimensions=1536) into a pre-existing 1536-dim pgvector column
# WHY-CORRECT: text-embedding-3-large natively returns 3072 dims but supports the `dimensions` parameter to shorten the vector; requesting dimensions=1536 yields a 1536-dim vector that fits an existing vector(1536) column. Documented, lossless-by-design downsizing.
# EXPECTED-WRONG: engine may claim a 3-large embedding is 3072-dim and therefore won't fit a 1536 column, missing the dimensions= override.
# CORRECT-VERDICT: no findings
"""Use text-embedding-3-large at 1536 dims to fit an existing pgvector column."""

import psycopg2
from openai import OpenAI

oai = OpenAI()
TARGET_DIM = 1536


def embed(text: str) -> list[float]:
    resp = oai.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=TARGET_DIM,
    )
    vector = resp.data[0].embedding
    assert len(vector) == TARGET_DIM
    return vector


def store(conn, doc_id: int, text: str) -> None:
    vector = embed(text)
    with conn.cursor() as cur:
        # documents.embedding is a vector(1536) column created earlier.
        cur.execute(
            "INSERT INTO documents (id, content, embedding) VALUES (%s, %s, %s)",
            (doc_id, text, vector),
        )
    conn.commit()


if __name__ == "__main__":
    connection = psycopg2.connect("dbname=app")
    store(connection, 1, "Vectors of differing native size can be reconciled.")
