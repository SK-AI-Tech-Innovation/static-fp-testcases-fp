# ACE-FP-EXPECT: clean
# CATEGORY: 31_prompt_caching_and_context
# SOURCE: Google Gen AI SDK (`google-genai`) explicit context caching (`caches.create`)
# WHY-CORRECT: Gemini explicit caching creates a named cache from large stable content, then
#              references it via `cached_content` on generation. TTL, model, and contents are
#              all set correctly. This is the documented `caches.create` flow.
# EXPECTED-WRONG: dated skill pack doesn't know `caches.create` / `cached_content` and flags
#                 "unknown API" or "you re-send the whole document every call".
# CORRECT-VERDICT: no findings
"""Create a Gemini explicit context cache and reuse it across generations."""
from google import genai
from google.genai import types

client = genai.Client()

LARGE_DOC = "Chapter contents.\n" * 5000


def build_cache() -> str:
    cache = client.caches.create(
        model="gemini-2.5-flash",
        config=types.CreateCachedContentConfig(
            display_name="handbook",
            system_instruction="Answer using only the cached handbook.",
            contents=[LARGE_DOC],
            ttl="3600s",
        ),
    )
    return cache.name


def ask(cache_name: str, question: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=types.GenerateContentConfig(cached_content=cache_name),
    )
    return response.text


if __name__ == "__main__":
    name = build_cache()
    print(ask(name, "Summarize chapter 3."))
