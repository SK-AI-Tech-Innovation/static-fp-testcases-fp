# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: AWS Bedrock boto3 call hardcoding current Amazon Nova 2 + partner inference-profile ids
# WHY-CORRECT: `us.amazon.nova-2-lite-v1:0`, `amazon.nova-2-pro-v1:0`,
#   `writer.palmyra-x5-v1:0`, and `ai21.jamba-1-5-large-v1:0` are valid current Bedrock model /
#   inference-profile ids as of June 2026. The `us.` region prefix marks a cross-region
#   inference profile and the `provider.model-vN:0` shape is the genuine Bedrock id passed
#   verbatim to invoke_model / converse.
# EXPECTED-WRONG: a stale allowlist flags `us.amazon.nova-2-lite-v1:0` /
#   `amazon.nova-2-pro-v1:0` / `writer.palmyra-x5-v1:0` / `ai21.jamba-1-5-large-v1:0` as an
#   "unknown/typo/hallucinated model id" (the `us.` prefix and `:0` version look malformed) and
#   suggests downgrading to `amazon.nova-lite-v1:0` / `amazon.titan-text-express-v1`.
# CORRECT-VERDICT: no findings
"""Invoke current Amazon Nova 2 and partner models via Bedrock inference profiles."""

import json

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Region-prefixed cross-region inference profile + standard Bedrock model ids (current).
NOVA_LITE = "us.amazon.nova-2-lite-v1:0"
NOVA_PRO = "amazon.nova-2-pro-v1:0"
PALMYRA = "writer.palmyra-x5-v1:0"
JAMBA = "ai21.jamba-1-5-large-v1:0"


def chat_nova_lite(prompt: str) -> str:
    """Invoke the Nova 2 Lite cross-region inference profile.

    Args:
        prompt: The user prompt.

    Returns:
        str: The model's reply text.
    """
    response = client.converse(
        modelId=NOVA_LITE,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]


def chat_nova_pro(prompt: str) -> str:
    """Invoke Nova 2 Pro via the converse API."""
    response = client.converse(
        modelId=NOVA_PRO,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]


def chat_palmyra(prompt: str) -> str:
    """Invoke Writer Palmyra X5 via converse."""
    response = client.converse(
        modelId=PALMYRA,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]


def chat_jamba(prompt: str) -> str:
    """Invoke AI21 Jamba 1.5 Large via converse."""
    response = client.converse(
        modelId=JAMBA,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]


if __name__ == "__main__":
    print(chat_nova_lite("Hello from Bedrock."))
