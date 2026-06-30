# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: openai-python — compatibility wrapper supporting BOTH v0.x and v1.x
# WHY-CORRECT: This is deliberate migration/compat code. The try/except probes for the v1
#   `OpenAI` client class; if present it uses client.chat.completions.create with attribute
#   access. If ImportError (v0.x installed), it falls back to openai.ChatCompletion.create
#   with dict access. BOTH branches are correct for their respective installed major version.
# EXPECTED-WRONG: engine may flag the v0 branch (ChatCompletion.create / dict access) as
#   deprecated, or call the dual-import try/except a "smelly compat shim", or want it removed.
# CORRECT-VERDICT: no findings
"""Wrapper that runs a chat completion on either openai v0.x or v1.x, whichever is installed."""
import os

try:
    from openai import OpenAI  # available only on openai>=1.0

    _V1_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    _IS_V1 = True
except ImportError:
    import openai  # openai<1.0 legacy module

    openai.api_key = os.environ["OPENAI_API_KEY"]
    _IS_V1 = False


def chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    messages = [{"role": "user", "content": prompt}]
    if _IS_V1:
        resp = _V1_CLIENT.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content
    # Legacy v0.x path: dict-like response access.
    resp = openai.ChatCompletion.create(model=model, messages=messages)
    return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(chat("Say hi in one word."))
