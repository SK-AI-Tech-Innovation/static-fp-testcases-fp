# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: AWS Bedrock Converse (boto3), verified June 2026
# WHY-CORRECT: boto3.client("bedrock-runtime").converse(...) is the unified inference API; modelId, messages with content lists, and inferenceConfig are all correct
# EXPECTED-WRONG: stale analyzer expects an OpenAI-style client.chat.completions.create and flags brt.converse(...) as an unknown method / malformed call
# CORRECT-VERDICT: no findings
"""Invoke a Bedrock model through the boto3 Converse API."""

import boto3

brt = boto3.client("bedrock-runtime", region_name="us-east-1")


def main() -> None:
    resp = brt.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        system=[{"text": "You are a helpful assistant."}],
        messages=[
            {"role": "user", "content": [{"text": "What is the tallest mountain?"}]}
        ],
        inferenceConfig={"maxTokens": 256, "topP": 0.9},
    )
    print(resp["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    main()
