# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: OpenAI Python SDK Responses API combining `tools=[...]` AND structured `text_format=Model`
# WHY-CORRECT: the Responses API supports tool calling and a typed final-output schema simultaneously:
#              the model may call the provided tool, then `responses.parse(..., text_format=Model)` returns
#              a validated `output_parsed`. Both features coexist; this is the current idiomatic pattern.
# EXPECTED-WRONG: skill examples only show `beta.chat.completions.parse(response_format=...)` without tools,
#                 so the engine either flags "not structured" (unfamiliar `text_format`) or claims tools and
#                 structured output are mutually exclusive and proposes an anachronistic downgrade.
# CORRECT-VERDICT: no findings
"""Look up weather via a tool and return a typed report using the OpenAI Responses API."""
from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


class WeatherReport(BaseModel):
    city: str = Field(description="City the report is for")
    temperature_c: float = Field(description="Temperature in Celsius")
    summary: str = Field(description="Short human-readable summary")


def weather_report(user_request: str) -> WeatherReport:
    response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Use the weather tool, then summarize."},
            {"role": "user", "content": user_request},
        ],
        tools=[WEATHER_TOOL],
        text_format=WeatherReport,
    )
    # Tools and text_format coexist: output_parsed is a validated WeatherReport.
    return response.output_parsed
