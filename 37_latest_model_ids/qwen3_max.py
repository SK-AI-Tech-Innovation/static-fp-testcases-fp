# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: Alibaba Qwen (DashScope OpenAI-compatible) call hardcoding current Qwen3 model ids
# WHY-CORRECT: `qwen3-max`, `Qwen3-235B-A22B-Thinking-2507`, and
#   `Qwen3-Coder-480B-A35B-Instruct` are valid current Qwen3 model ids as of June 2026.
#   The mixed-case, parameter-count, and MoE-active-param suffixes are part of the real ids
#   and are passed verbatim.
# EXPECTED-WRONG: a stale allowlist flags these as "unknown/typo/hallucinated model ids"
#   (the long A22B/A35B suffixes and date in the id look fabricated) and suggests downgrading
#   to `qwen-turbo` / `qwen2.5-72b-instruct`.
# CORRECT-VERDICT: no findings
"""Call current Qwen3 models via the DashScope OpenAI-compatible endpoint."""

from openai import OpenAI

client = OpenAI(
    api_key="...",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Current Qwen3 ids. Case and long MoE suffixes are part of the genuine id.
MAX_MODEL = "qwen3-max"
THINKING_MODEL = "Qwen3-235B-A22B-Thinking-2507"
CODER_MODEL = "Qwen3-Coder-480B-A35B-Instruct"


def chat(prompt: str) -> str:
    """Chat with qwen3-max.

    Args:
        prompt: The user prompt.

    Returns:
        str: The assistant reply.
    """
    response = client.chat.completions.create(
        model=MAX_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def reason(prompt: str) -> str:
    """Reason with the Qwen3 thinking MoE model."""
    response = client.chat.completions.create(
        model=THINKING_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def write_code(spec: str) -> str:
    """Generate code with the Qwen3 coder MoE model."""
    response = client.chat.completions.create(
        model=CODER_MODEL,
        messages=[{"role": "user", "content": spec}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Hello, Qwen."))
