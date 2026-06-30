# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: LangChain (@tool decorator with a Pydantic args_schema, bound to a model)
# WHY-CORRECT: The tool has a typed args schema (Pydantic model with Field
#   descriptions and a constraint), a clear docstring used as the tool
#   description, and is bound to the chat model via bind_tools so the model can
#   discover and call it. This is idiomatic LangChain — nothing is missing.
# EXPECTED-WRONG: engine may suggest "add an args schema", "document the tool",
#   or "bind the tool to the model" — all already present.
# CORRECT-VERDICT: no findings
"""Idiomatic LangChain @tool with a typed args schema, bound to a model."""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ConvertArgs(BaseModel):
    """Arguments for the temperature conversion tool."""

    celsius: float = Field(
        description="Temperature in degrees Celsius to convert to Fahrenheit.",
        ge=-273.15,  # physical lower bound: absolute zero
    )


@tool(args_schema=ConvertArgs)
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit.

    Use this whenever the user gives a Celsius value and wants it in Fahrenheit.
    """
    return celsius * 9.0 / 5.0 + 32.0


model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools([celsius_to_fahrenheit])


def ask(question: str):
    """Send a question to a model that can call the conversion tool."""
    return model_with_tools.invoke(question)


if __name__ == "__main__":
    result = ask("Convert 100 Celsius to Fahrenheit.")
    print(result.tool_calls)
