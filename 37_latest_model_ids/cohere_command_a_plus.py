# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: Cohere Python SDK call hardcoding current Command / embed / rerank model ids
# WHY-CORRECT: `command-a-plus-05-2026`, `embed-v4.0`, and `rerank-v4.0-pro` are valid current
#   Cohere model ids as of June 2026. The dated `05-2026` snapshot, `v4.0`, and `-pro` suffixes
#   are part of the genuine id and are passed verbatim.
# EXPECTED-WRONG: a stale allowlist flags `command-a-plus-05-2026` / `embed-v4.0` /
#   `rerank-v4.0-pro` as an "unknown/typo/hallucinated model id" and suggests downgrading to
#   `command-r-plus` / `embed-english-v3.0` / `rerank-english-v3.0`.
# CORRECT-VERDICT: no findings
"""Call current Cohere chat, embed, and rerank models with their pinned ids."""

import cohere

client = cohere.ClientV2(api_key="...")

# Current Cohere ids.
CHAT_MODEL = "command-a-plus-05-2026"
EMBED_MODEL = "embed-v4.0"
RERANK_MODEL = "rerank-v4.0-pro"


def chat(prompt: str) -> str:
    """Chat with command-a-plus-05-2026.

    Args:
        prompt: The user prompt.

    Returns:
        str: The assistant reply text.
    """
    response = client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content[0].text


def embed(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` with embed-v4.0."""
    response = client.embed(
        model=EMBED_MODEL,
        texts=texts,
        input_type="search_document",
        embedding_types=["float"],
    )
    return response.embeddings.float_


def rerank(query: str, documents: list[str]):
    """Rerank ``documents`` against ``query`` with rerank-v4.0-pro."""
    return client.rerank(model=RERANK_MODEL, query=query, documents=documents)


if __name__ == "__main__":
    print(chat("Hello, Command."))
