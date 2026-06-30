# ACE-FP-EXPECT: clean
# CATEGORY: 29_finetuning_and_batch
# SOURCE: OpenAI SDK files.create for retrieval/processing
# WHY-CORRECT: uploading a document with purpose="assistants" via client.files.create is the documented way to make a file available for retrieval/processing. File upload is a storage operation, not a chat call.
# EXPECTED-WRONG: a chat-focused engine may flag files.create / the "assistants" purpose as an unrecognized operation and emit an irrelevant best-practice finding about a missing model or messages.
"""Upload documents for retrieval/processing and return their file ids."""

import os
from typing import Iterable

from openai import OpenAI


def upload_documents(paths: Iterable[str]) -> list[str]:
    """Upload each document for retrieval; return the resulting file ids."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    file_ids: list[str] = []

    for path in paths:
        with open(path, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="assistants")
        file_ids.append(uploaded.id)

    return file_ids


if __name__ == "__main__":
    ids = upload_documents(["docs/handbook.pdf", "docs/policy.pdf"])
    print("Uploaded file ids:", ids)
