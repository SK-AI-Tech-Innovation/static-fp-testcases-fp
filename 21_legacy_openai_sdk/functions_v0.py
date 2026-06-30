# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: openai-python v0.x (e.g. 0.28) — authentic legacy API
# WHY-CORRECT: Function calling in v0.x used the `functions=[...]` and `function_call="auto"`
#   kwargs (this predates the `tools=`/`tool_calls` schema introduced later). Reading the
#   model's chosen call from response['choices'][0]['message']['function_call'] is the
#   documented v0.x pattern. Providing a JSON-schema parameters block was best practice.
# EXPECTED-WRONG: engine may flag `functions=`/`function_call=` as deprecated, mis-fix to the
#   `tools=[{"type": "function", ...}]` schema, or call the function_call dict access a bug.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy openai v0.x function calling via the functions= / function_call= kwargs."""
import json
import os

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

FUNCTIONS = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]


def ask_weather(question: str) -> dict:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": question}],
        functions=FUNCTIONS,
        function_call="auto",
    )
    message = response["choices"][0]["message"]
    if message.get("function_call"):
        return json.loads(message["function_call"]["arguments"])
    return {"content": message["content"]}


if __name__ == "__main__":
    print(ask_weather("What's the weather like in Boston?"))
