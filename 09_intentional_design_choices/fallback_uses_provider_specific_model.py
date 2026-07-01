# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# LANGUAGE: python
# SOURCE: ai-readable-data PR #1336 (llm_client.py) — ACE: "fallback 재시도 시 primary와 다른 model 사용 가능성"; author confirmed FP (intentional per-provider model)
# WHY-CORRECT: the fallback path deliberately uses a provider-specific model id (the Azure deployment
#              name, e.g. gpt-5.4) instead of the primary vLLM model name. Sending the primary model
#              name to the fallback provider would FAIL — different providers require different model
#              identifiers. Using a different model on the fallback is correct by design, not a bug.
# EXPECTED-WRONG: engine sees the fallback call passing a different model than the primary and flags a
#                 "primary/fallback model mismatch bug", not understanding that each provider needs its
#                 own model id.
# CORRECT-VERDICT: no findings
"""Primary/fallback LLM calls that intentionally use provider-specific model ids.

ACE flagged the fallback using a different model than the primary. That is required: the
primary (vLLM) and the fallback (Azure) each need their own model identifier.
"""
import os

from openai import OpenAI

# Primary: in-house vLLM (its own model name). Fallback: Azure (its own deployment name).
PRIMARY_MODEL = os.environ.get("LLM_MODEL", "vllm-qwen3")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gpt-5.4")  # Azure deployment name

_primary = OpenAI(base_url=os.environ["VLLM_URL"], api_key="dummy")
_fallback = OpenAI(api_key=os.environ["AZURE_API_KEY"])


def chat_with_fallback(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        r = _primary.chat.completions.create(model=PRIMARY_MODEL, messages=messages)
    except Exception:
        # Fallback MUST use the Azure model id — sending PRIMARY_MODEL here would fail.
        r = _fallback.chat.completions.create(model=FALLBACK_MODEL, messages=messages)
    return r.choices[0].message.content
