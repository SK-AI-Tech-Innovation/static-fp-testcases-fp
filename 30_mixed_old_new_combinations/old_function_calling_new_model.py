# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python v1.x Chat Completions using the legacy functions=/function_call= parameters with a current model gpt-4o
# WHY-CORRECT: the deprecated `functions` and `function_call` parameters are still accepted by the chat.completions endpoint and return message.function_call. They remain supported (not removed) and work with current models like gpt-4o.
# EXPECTED-WRONG: engine may insist functions= is removed and demand tools=/tool_calls=, or claim a new model rejects the legacy function-calling shape.
# CORRECT-VERDICT: no findings
"""Use legacy functions= function calling with the current gpt-4o model."""

import json

from openai import OpenAI

client = OpenAI()

FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def ask(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        functions=FUNCTIONS,
        function_call="auto",
    )
    message = response.choices[0].message
    if message.function_call is not None:
        return message.function_call.name, json.loads(message.function_call.arguments)
    return None, message.content


if __name__ == "__main__":
    print(ask("What's the weather in Seoul?"))
