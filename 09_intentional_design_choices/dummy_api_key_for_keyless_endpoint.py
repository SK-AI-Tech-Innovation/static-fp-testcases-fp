# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# LANGUAGE: python
# SOURCE: ai-readable-data PR #817 (embeddings.py) — ACE: "api_key None일 때 'dummy' 하드코딩 (인증 혼동)"; author confirmed FP (intentional)
# STANDARD: CWE-798/CWE-259 apply only when a credential participates in real authentication; a dummy key sent to a keyless/anonymous endpoint authenticates to nothing — not a hardcoded-credential vuln
# WHY-CORRECT: passing api_key="dummy" is a deliberate choice for keyless endpoints (in-house vLLM /
#              TEI bge-m3). OpenAIEmbeddings silently falls back to the global OPENAI_API_KEY env when
#              api_key is None; passing an explicit placeholder PREVENTS that accidental global-key use.
#              The behavior is locked by a regression test.
# EXPECTED-WRONG: engine flags the literal "dummy" key as a hardcoded-credential / auth-confusion bug,
#                 missing that it is an intentional guard against silent env-key fallback for keyless backends.
# CORRECT-VERDICT: no findings
"""Pass an explicit placeholder key to a keyless embeddings endpoint on purpose.

ACE flagged the "dummy" api_key. For vLLM/TEI (no key required) this explicit placeholder
deliberately blocks OpenAIEmbeddings' silent fallback to the global OPENAI_API_KEY env.
"""
from typing import Any


def build_embeddings(provider: str, api_key: str | None, base_url: str) -> Any:
    from langchain_openai import OpenAIEmbeddings

    # Keyless backends (vLLM / TEI) need no real key. Passing "dummy" explicitly stops
    # OpenAIEmbeddings from silently using the global OPENAI_API_KEY env when api_key is None.
    return OpenAIEmbeddings(
        model="bge-m3",
        api_key=api_key or "dummy",
        base_url=base_url,
        chunk_size=64,
    )
