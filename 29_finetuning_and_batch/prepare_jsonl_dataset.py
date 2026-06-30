# ACE-FP-EXPECT: clean
# CATEGORY: 29_finetuning_and_batch
# SOURCE: OpenAI fine-tune dataset preparation (JSONL messages format)
# WHY-CORRECT: Each JSONL line is a {"messages": [...]} object with system/user/assistant turns, exactly the schema the fine-tuning API expects. This is pure data prep with no LLM call.
# EXPECTED-WRONG: engine may misread the inline messages list as a live chat request and flag a "missing model" or "no API call" best-practice issue.
"""Build a JSONL fine-tuning dataset in the chat messages format."""

import json
from typing import Iterable


def to_example(system: str, user: str, assistant: str) -> dict:
    """Construct one fine-tune example in the messages schema."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def write_dataset(rows: Iterable[tuple[str, str, str]], out_path: str) -> int:
    """Write examples to a JSONL file; return the number of lines written."""
    count = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for system, user, assistant in rows:
            example = to_example(system, user, assistant)
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    sample = [
        ("You are a support agent.", "How do I reset my password?", "Click 'Forgot password'."),
        ("You are a support agent.", "Where are my invoices?", "Open Billing > Invoices."),
    ]
    print("Wrote", write_dataset(sample, "data/train.jsonl"), "examples")
