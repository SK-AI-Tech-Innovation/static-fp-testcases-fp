# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: AWS Bedrock Converse (boto3), verified June 2026
# WHY-CORRECT: Converse takes system as a TOP-LEVEL list of {"text":...} blocks (system=[{"text":...}]), not as a {"role":"system"} message in the messages array
# EXPECTED-WRONG: stale analyzer expects a system role inside messages and flags the top-level system=[{"text":...}] list as a malformed/misplaced system prompt
# CORRECT-VERDICT: no findings
"""Pass a system prompt to Bedrock Converse via the top-level system list."""

import boto3

brt = boto3.client("bedrock-runtime", region_name="us-east-1")


def main() -> None:
    resp = brt.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        system=[{"text": "You are a terse assistant. Answer in one sentence."}],
        messages=[
            {"role": "user", "content": [{"text": "Why is the sky blue?"}]}
        ],
        inferenceConfig={"maxTokens": 256, "topP": 0.9},
    )

    print(resp["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    main()
