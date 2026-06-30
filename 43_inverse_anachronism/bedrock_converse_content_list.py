# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: AWS Bedrock Converse API via boto3 (`bedrock-runtime.converse`).
# WHY-CORRECT: the Bedrock Converse API requires message `content` to be a LIST of content blocks, e.g.
#              `[{"text": "..."}]`, and the response text lives at
#              `response["output"]["message"]["content"][0]["text"]`. The content-list shape is mandatory;
#              this code is correct.
# EXPECTED-WRONG: a stale engine "knows" the OpenAI/plain-string convention where `content` is a bare string
#                 and "fixes" the Converse call by replacing `content=[{"text": prompt}]` with
#                 `content=prompt` (and reading a string response). boto3 validates the Converse schema and
#                 rejects a plain string for `content` → ParamValidationError. The "fix" breaks the call.
# CORRECT-VERDICT: no findings — keep the content-block lists (`[{"text": ...}]`). Do not collapse them to
#                  plain strings.
"""Bedrock Converse with boto3 using content-block lists — not plain strings, by design."""
from __future__ import annotations

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "anthropic.claude-opus-4-8"


def chat(prompt: str) -> str:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        system=[{"text": "You are a helpful assistant."}],
        inferenceConfig={"maxTokens": 1024},
    )
    return response["output"]["message"]["content"][0]["text"]
