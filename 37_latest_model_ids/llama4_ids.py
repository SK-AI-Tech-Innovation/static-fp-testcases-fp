# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: Hugging Face / vLLM-style call hardcoding current Llama 4 repo ids
# WHY-CORRECT: `meta-llama/Llama-4-Scout-17B-16E-Instruct` and
#   `meta-llama/Llama-4-Maverick-17B-128E-Instruct` are valid current Meta Llama 4 model repo
#   ids as of June 2026. The org-prefix, parameter count, and expert-count (16E/128E) suffixes
#   are part of the genuine repo id and are passed verbatim.
# EXPECTED-WRONG: a stale allowlist flags these as "unknown/typo/hallucinated model ids"
#   (the 16E/128E expert suffix looks invented) and suggests downgrading to
#   `meta-llama/Llama-3-8B-Instruct`.
# CORRECT-VERDICT: no findings
"""Load current Llama 4 models by their canonical Hugging Face repo ids."""

from transformers import pipeline

# Current Meta Llama 4 repo ids (org-prefixed; NxE = number of experts).
SCOUT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MAVERICK_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"


def build_scout():
    """Build a text-generation pipeline on Llama 4 Scout.

    Returns:
        The instantiated Hugging Face pipeline.
    """
    return pipeline("text-generation", model=SCOUT_MODEL)


def build_maverick():
    """Build a text-generation pipeline on Llama 4 Maverick."""
    return pipeline("text-generation", model=MAVERICK_MODEL)


if __name__ == "__main__":
    gen = build_scout()
    print(gen("Hello", max_new_tokens=16))
