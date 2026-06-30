# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: AWS Bedrock Runtime Converse API (`bedrock-runtime.converse(..., toolConfig=...)`)
# WHY-CORRECT: `converse` is the current unified Bedrock chat API; tool use is declared via `toolConfig`
#              and tool results are returned as a `toolResult` content block. The loop handles
#              `stopReason == "tool_use"` correctly and reads typed `toolUse` input.
# EXPECTED-WRONG: a Python+OpenAI-centric engine doesn't recognize boto3's Converse shape; it may flag
#                 "no structured output", "raw/unparsed completion", or want an OpenAI-style response_format.
# CORRECT-VERDICT: no findings
"""AWS Bedrock Converse API tool-use loop with a currency-conversion tool."""
from __future__ import annotations

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "convert_currency",
                "description": "Convert an amount between two ISO currency codes.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number"},
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                        },
                        "required": ["amount", "from", "to"],
                    }
                },
            }
        }
    ]
}


def _convert(amount: float, from_: str, to: str) -> float:
    rate = 1.08  # stand-in for a live FX rate lookup
    return round(amount * rate, 2)


def ask(prompt: str) -> str:
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    while True:
        resp = client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=messages,
            toolConfig=TOOL_CONFIG,
        )
        output = resp["output"]["message"]
        messages.append(output)

        if resp["stopReason"] != "tool_use":
            return "".join(b.get("text", "") for b in output["content"])

        tool_results = []
        for block in output["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            value = _convert(
                tool_use["input"]["amount"],
                tool_use["input"]["from"],
                tool_use["input"]["to"],
            )
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": {"converted": value}}],
                    }
                }
            )
        messages.append({"role": "user", "content": tool_results})
