# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: LangChain `with_structured_output(TypedDict)` bound to a `TypedDict` schema
# WHY-CORRECT: structured output accepts a `TypedDict` (with `Annotated` field descriptions) just like a
#              Pydantic model; the runnable returns a dict conforming to the typed schema. The schema is
#              enforced by the structured-output layer — no manual JSON parsing on the consumer side.
# EXPECTED-WRONG: engine only recognizes a Pydantic `BaseModel` + `response_format=`/`.parse(...)` as
#                 structured output and flags the `TypedDict`-bound call as "not structured / free-text".
# CORRECT-VERDICT: no findings
"""Extract typed weather data bound to a TypedDict schema via LangChain structured output."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI


class WeatherData(TypedDict):
    """Structured weather fields the model must populate."""

    city: Annotated[str, ..., "City the data is for"]
    temperature_c: Annotated[float, ..., "Temperature in Celsius"]
    condition: Annotated[str, ..., "Short weather description, e.g. 'cloudy'"]


_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
_extractor = _llm.with_structured_output(WeatherData)


def extract_weather(text: str) -> WeatherData:
    # Returns a dict conforming to the WeatherData TypedDict — schema enforced, no json.loads.
    return _extractor.invoke(
        [
            ("system", "Extract the weather fields into the schema."),
            ("human", text),
        ]
    )
