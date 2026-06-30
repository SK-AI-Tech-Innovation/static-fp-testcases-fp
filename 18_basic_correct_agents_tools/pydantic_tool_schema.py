# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: Anthropic Python SDK tool definition + Pydantic v2 model schema
# WHY-CORRECT: The tool's input schema is generated from a Pydantic model whose
#   every field has a Field(description=...) and meaningful constraints (ge/le
#   ranges, enum via Literal, max_length). model_json_schema() produces a fully
#   described, constrained JSON Schema. There is genuinely nothing to flag.
# EXPECTED-WRONG: engine may suggest "describe the tool fields" or "add
#   constraints / validation to the parameters" — every field already has both.
# CORRECT-VERDICT: no findings
"""A tool whose args are a Pydantic model with Field descriptions + constraints."""

from typing import Literal

from pydantic import BaseModel, Field


class SearchFlightsArgs(BaseModel):
    """Validated arguments for the flight-search tool."""

    destination: str = Field(
        description="IATA airport or city name for the destination.",
        max_length=64,
    )
    passengers: int = Field(
        description="Number of passengers on the booking.",
        ge=1,
        le=9,
    )
    cabin: Literal["economy", "premium", "business", "first"] = Field(
        default="economy",
        description="Cabin class to search for.",
    )


SEARCH_FLIGHTS_TOOL = {
    "name": "search_flights",
    "description": (
        "Search available flights. Call this when the user asks to find or book "
        "a flight to a destination for a given number of passengers."
    ),
    "input_schema": SearchFlightsArgs.model_json_schema(),
}


def search_flights(args: SearchFlightsArgs) -> list[dict]:
    """Run the flight search against validated, typed arguments."""
    return [
        {
            "destination": args.destination,
            "passengers": args.passengers,
            "cabin": args.cabin,
            "price_usd": 420,
        }
    ]


if __name__ == "__main__":
    validated = SearchFlightsArgs(destination="Tokyo", passengers=2, cabin="business")
    print(search_flights(validated))
    print(SEARCH_FLIGHTS_TOOL["input_schema"]["properties"]["passengers"])
