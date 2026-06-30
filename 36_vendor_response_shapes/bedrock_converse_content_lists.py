# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: AWS Bedrock Converse (boto3), verified June 2026
# WHY-CORRECT: Converse messages use content as a list of typed parts [{"text":...}], and the reply text is resp["output"]["message"]["content"][0]["text"], NOT resp.choices[0].message.content
# EXPECTED-WRONG: stale analyzer expects .choices / a plain string content and flags the content-list shape and dict-indexed output access as malformed
# CORRECT-VERDICT: no findings
"""Call AWS Bedrock Converse and read the content-list response shape."""

import boto3

brt = boto3.client("bedrock-runtime", region_name="us-east-1")


def main() -> None:
    resp = brt.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[
            {"role": "user", "content": [{"text": "Name three primary colors."}]}
        ],
        inferenceConfig={"maxTokens": 256, "topP": 0.9},
    )

    # Converse output is nested message content blocks.
    print(resp["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    main()
