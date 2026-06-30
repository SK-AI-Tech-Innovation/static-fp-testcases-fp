# ACE-FP-EXPECT: clean
# CATEGORY: 29_finetuning_and_batch
# SOURCE: OpenAI SDK fine_tuning.jobs + files
# WHY-CORRECT: Uploading the training file with purpose="fine-tune", then calling client.fine_tuning.jobs.create(training_file=file.id, model="gpt-4o-mini-2024-07-18") is the documented fine-tuning flow. No chat call is involved.
# EXPECTED-WRONG: a text-chat-oriented engine may expect chat.completions.create and flag the fine_tuning.jobs / files.create calls as unknown or "wrong API" usage.
"""Create an OpenAI fine-tuning job from an uploaded JSONL training file."""

import os

from openai import OpenAI


def submit_finetune(training_path: str, base_model: str = "gpt-4o-mini-2024-07-18") -> str:
    """Upload a training file and start a fine-tuning job; return the job id."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(training_path, "rb") as fh:
        training_file = client.files.create(file=fh, purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model=base_model,
        suffix="support-bot",
    )
    return job.id


if __name__ == "__main__":
    print("Started job:", submit_finetune("data/train.jsonl"))
