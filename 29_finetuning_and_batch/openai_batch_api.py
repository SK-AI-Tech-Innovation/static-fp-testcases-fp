# ACE-FP-EXPECT: clean
# CATEGORY: 29_finetuning_and_batch
# SOURCE: OpenAI SDK batches API
# WHY-CORRECT: Uploading a JSONL with purpose="batch" and calling client.batches.create(input_file_id=..., endpoint="/v1/chat/completions", completion_window="24h") is the exact documented Batch API contract.
# EXPECTED-WRONG: engine may flag batches.create or the "/v1/chat/completions" endpoint string / "24h" window as an unrecognized call and emit an irrelevant best-practice finding.
"""Submit an asynchronous OpenAI Batch job over a JSONL request file."""

import os

from openai import OpenAI


def submit_batch(requests_path: str) -> str:
    """Upload a batch input file and create a batch job; return the batch id."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(requests_path, "rb") as fh:
        batch_input = client.files.create(file=fh, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_input.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly classification run"},
    )
    return batch.id


if __name__ == "__main__":
    print("Created batch:", submit_batch("data/batch_requests.jsonl"))
