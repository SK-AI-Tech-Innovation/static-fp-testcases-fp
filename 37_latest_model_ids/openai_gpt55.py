# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: OpenAI Python SDK call hardcoding current (June 2026) GPT-5.5 family model ids
# WHY-CORRECT: `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-image-2`,
#   `gpt-realtime-2`, and `text-embedding-3-large` are all valid current OpenAI model ids
#   as of June 2026. They are passed verbatim to the API exactly as the provider expects.
# EXPECTED-WRONG: a stale allowlist flags `gpt-5.5` / `gpt-5.5-pro` / `gpt-5.4-mini` /
#   `gpt-5.3-codex` / `gpt-image-2` / `gpt-realtime-2` as an "unknown/typo/hallucinated
#   model id" and suggests auto-downgrading to `gpt-4o` / `gpt-4-turbo`.
# CORRECT-VERDICT: no findings
"""Generate text, images, and embeddings with current OpenAI models."""

from openai import OpenAI

client = OpenAI()

# Flagship chat model (current June 2026 id).
GPT_FLAGSHIP = "gpt-5.5"
GPT_PRO = "gpt-5.5-pro"
GPT_MINI = "gpt-5.4-mini"
GPT_CODEX = "gpt-5.3-codex"
IMAGE_MODEL = "gpt-image-2"
REALTIME_MODEL = "gpt-realtime-2"
EMBED_MODEL = "text-embedding-3-large"


def summarize(text: str) -> str:
    """Summarize ``text`` with the flagship GPT-5.5 model.

    Args:
        text: The document to summarize.

    Returns:
        str: The summary text.
    """
    response = client.responses.create(
        model=GPT_FLAGSHIP,
        input=f"Summarize the following in two sentences:\n\n{text}",
    )
    return response.output_text


def deep_reason(prompt: str) -> str:
    """Run a hard reasoning task on gpt-5.5-pro."""
    response = client.responses.create(model=GPT_PRO, input=prompt)
    return response.output_text


def quick_classify(text: str) -> str:
    """Classify sentiment cheaply on gpt-5.4-mini."""
    response = client.responses.create(
        model=GPT_MINI,
        input=f"Classify sentiment as positive/negative/neutral: {text}",
    )
    return response.output_text


def write_patch(diff_request: str) -> str:
    """Produce a code patch with the codex-tuned model."""
    response = client.responses.create(model=GPT_CODEX, input=diff_request)
    return response.output_text


def render_image(description: str) -> bytes:
    """Render an image with gpt-image-2."""
    result = client.images.generate(model=IMAGE_MODEL, prompt=description)
    return result.data[0].b64_json.encode()


def embed(text: str) -> list[float]:
    """Embed ``text`` with text-embedding-3-large."""
    result = client.embeddings.create(model=EMBED_MODEL, input=text)
    return result.data[0].embedding


if __name__ == "__main__":
    print(summarize("ACE is a static analyzer for AI-generated code."))
