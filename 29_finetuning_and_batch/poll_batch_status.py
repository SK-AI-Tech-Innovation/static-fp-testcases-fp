# ACE-FP-EXPECT: clean
# CATEGORY: 29_finetuning_and_batch
# SOURCE: OpenAI SDK batches.retrieve polling loop
# WHY-CORRECT: polling client.batches.retrieve(batch_id) until status is terminal, with capped exponential backoff between checks, is the correct way to wait on an async batch job. Terminal statuses ("completed"/"failed"/"expired"/"cancelled") are handled explicitly.
# EXPECTED-WRONG: engine may flag the backoff sleep loop or the batches.retrieve status polling as a busy-wait anti-pattern or unknown API, producing an irrelevant finding.
"""Poll an OpenAI batch job to completion with capped exponential backoff."""

import os
import time

from openai import OpenAI

_TERMINAL = {"completed", "failed", "expired", "cancelled"}
_MAX_DELAY = 60.0


def wait_for_batch(batch_id: str) -> str:
    """Block until the batch reaches a terminal status; return that status."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    delay = 2.0

    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in _TERMINAL:
            return batch.status
        time.sleep(delay)
        delay = min(delay * 2, _MAX_DELAY)


if __name__ == "__main__":
    final_status = wait_for_batch("batch_abc123")
    print("Batch finished with status:", final_status)
