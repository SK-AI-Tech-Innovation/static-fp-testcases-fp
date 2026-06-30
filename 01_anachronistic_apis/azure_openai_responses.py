# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Azure OpenAI using the Responses API surface (`AzureOpenAI().responses.parse`)
# WHY-CORRECT: `AzureOpenAI` shares the Responses API with the public SDK; `responses.parse(text_format=Model)`
#              returns a validated `output_parsed`. Azure addresses models by deployment name and requires an
#              `api_version`/`azure_endpoint` — this is the documented, current Azure structured-output path.
# EXPECTED-WRONG: engine doesn't recognize the `AzureOpenAI` client or the deployment-name "model", and either
#                 flags the config as wrong or claims structured output is missing because it isn't the dated
#                 `beta.chat.completions.parse(response_format=...)` example.
# CORRECT-VERDICT: no findings
"""Extract a typed meeting summary using Azure OpenAI's Responses API."""
from __future__ import annotations

import os

from openai import AzureOpenAI
from pydantic import BaseModel, Field

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-03-01-preview",
)


class MeetingSummary(BaseModel):
    title: str = Field(description="Short meeting title")
    decisions: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


def summarize_meeting(transcript: str) -> MeetingSummary:
    response = client.responses.parse(
        model="gpt-4.1",  # Azure deployment name
        input=[
            {"role": "system", "content": "Summarize the meeting into the schema."},
            {"role": "user", "content": transcript},
        ],
        text_format=MeetingSummary,
    )
    # output_parsed is a validated MeetingSummary — no manual JSON parsing.
    return response.output_parsed
