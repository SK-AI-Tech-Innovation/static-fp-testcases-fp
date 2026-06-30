# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: tiktoken + openai-python v0.x era — authentic legacy token counting
# WHY-CORRECT: tiktoken.encoding_for_model("gpt-3.5-turbo") returns the cl100k_base encoding,
#   and the per-message token-counting helper (3 tokens of overhead per message, +3 priming
#   tokens for the reply) matches OpenAI's published cookbook formula for the 0613 chat models.
#   The model is old but the counting is exactly correct for it.
# EXPECTED-WRONG: engine may flag the old model name and push a newer model/encoding, claim the
#   per-message overhead constants are a bug, or call encoding_for_model deprecated.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Count chat tokens for a legacy gpt-3.5-turbo model using tiktoken."""
from typing import Dict, List

import tiktoken


def num_tokens_from_messages(messages: List[Dict[str, str]],
                             model: str = "gpt-3.5-turbo-0613") -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


if __name__ == "__main__":
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    print(num_tokens_from_messages(msgs))
