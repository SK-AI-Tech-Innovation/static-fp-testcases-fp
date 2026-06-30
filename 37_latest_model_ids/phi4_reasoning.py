# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: Hugging Face transformers call hardcoding the current Microsoft Phi-4 reasoning repo id
# WHY-CORRECT: `microsoft/Phi-4-reasoning` is a valid current Microsoft Phi-4 model repo id as
#   of June 2026. The org prefix and `-reasoning` suffix are part of the genuine repo id and are
#   passed verbatim to from_pretrained.
# EXPECTED-WRONG: a stale allowlist flags `microsoft/Phi-4-reasoning` as an
#   "unknown/typo/hallucinated model id" and suggests downgrading to `microsoft/phi-2` /
#   `microsoft/Phi-3-mini-4k-instruct`.
# CORRECT-VERDICT: no findings
"""Load the current Microsoft Phi-4 reasoning model by its Hugging Face repo id."""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Current Microsoft Phi-4 reasoning repo id.
PHI4_REASONING = "microsoft/Phi-4-reasoning"


def load():
    """Load the Phi-4 reasoning tokenizer and model.

    Returns:
        tuple: The loaded ``(tokenizer, model)`` pair.
    """
    tokenizer = AutoTokenizer.from_pretrained(PHI4_REASONING)
    model = AutoModelForCausalLM.from_pretrained(PHI4_REASONING)
    return tokenizer, model


if __name__ == "__main__":
    tok, mdl = load()
    print(type(mdl).__name__)
