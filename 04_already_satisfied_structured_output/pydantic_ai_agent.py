# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: `pydantic_ai.Agent(output_type=Model)` returning a validated result
# WHY-CORRECT: PydanticAI validates the model output against the typed output_type and exposes
#              it as `result.output`; invalid output triggers an automatic re-ask, so the
#              consumer never parses free text
# EXPECTED-WRONG: engine flags "missing structured output / parsing raw model text" because
#                 there is no `response_format=` or `.with_structured_output(...)` token
# CORRECT-VERDICT: no findings
"""Summarize a weather query into a typed forecast with a PydanticAI agent."""
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class Forecast(BaseModel):
    city: str = Field(description="City the forecast is for")
    temperature_c: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Short weather description, e.g. 'sunny'")


weather_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=Forecast,
    system_prompt="Produce a structured weather forecast for the requested city.",
)


async def forecast_for(query: str) -> Forecast:
    result = await weather_agent.run(query)
    # result.output is a validated Forecast instance, never a raw JSON string.
    return result.output
