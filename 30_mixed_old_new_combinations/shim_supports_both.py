# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python (version-detecting shim dispatching to v0 ChatCompletion or v1 client)
# WHY-CORRECT: the shim inspects the installed openai version and routes to the matching API surface — legacy openai.ChatCompletion.create on v0, the OpenAI() client on v1. Each branch only runs against the SDK that defines those symbols, so both paths are valid where reached.
# EXPECTED-WRONG: engine may flag whichever branch doesn't match the assumed installed version as "API does not exist", not realizing the version guard gates it.
# CORRECT-VERDICT: no findings
"""Version-aware shim that dispatches to the old or new OpenAI SDK surface."""

from importlib.metadata import version

import openai

_MAJOR = int(version("openai").split(".")[0])


def chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    messages = [{"role": "user", "content": prompt}]

    if _MAJOR >= 1:
        client = openai.OpenAI()
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content

    # Legacy v0 surface.
    resp = openai.ChatCompletion.create(model=model, messages=messages)
    return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(chat("Say hello in Spanish."))
