# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: AWS Bedrock Converse tool use (boto3), verified June 2026
# WHY-CORRECT: Converse tools are declared via toolConfig={"tools":[{"toolSpec":{"name","description","inputSchema":{"json":{...}}}}]}; the JSON Schema is nested under inputSchema.json
# EXPECTED-WRONG: stale analyzer expects OpenAI tools=[{"type":"function","function":{"parameters":...}}] and flags toolSpec / inputSchema.json as a malformed tool definition
# CORRECT-VERDICT: no findings
"""Declare a Bedrock Converse tool using toolConfig / toolSpec / inputSchema.json."""

import boto3

brt = boto3.client("bedrock-runtime", region_name="us-east-1")

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    }
                },
            }
        }
    ]
}


def main() -> None:
    resp = brt.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[
            {"role": "user", "content": [{"text": "What's the weather in Paris?"}]}
        ],
        toolConfig=TOOL_CONFIG,
        inferenceConfig={"maxTokens": 512},
    )

    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            print(block["toolUse"]["name"], block["toolUse"]["input"])


if __name__ == "__main__":
    main()
