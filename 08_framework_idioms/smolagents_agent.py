# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: HuggingFace smolagents CodeAgent + ToolCallingAgent with an @tool function
# WHY-CORRECT: This is idiomatic smolagents: a Python function decorated with @tool (typed
#              args + docstring become the schema), passed into a CodeAgent (which writes
#              and runs code) and a ToolCallingAgent (which emits JSON tool calls), both
#              driven by an InferenceClientModel. The framework generates and executes the
#              tool-calling loop; there is no manual prompt or raw HTTP call.
# EXPECTED-WRONG: CodeAgent executing model-written Python may be flagged as "arbitrary code
#                 execution / unsafe"; the @tool function has no explicit JSON schema literal,
#                 so schema checks may report "tool missing parameters".
# CORRECT-VERDICT: no findings
"""Idiomatic HuggingFace smolagents: @tool function with CodeAgent and ToolCallingAgent."""
from __future__ import annotations

from smolagents import CodeAgent, InferenceClientModel, ToolCallingAgent, tool


@tool
def unit_price(total: float, quantity: int) -> float:
    """Compute the price per unit.

    Args:
        total: The total amount paid.
        quantity: The number of units purchased (must be positive).
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    return total / quantity


model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

code_agent = CodeAgent(tools=[unit_price], model=model, max_steps=4)
tool_calling_agent = ToolCallingAgent(tools=[unit_price], model=model, max_steps=4)


def run_with_code_agent(task: str) -> str:
    """Let the CodeAgent solve a task by writing and executing Python."""
    return str(code_agent.run(task))


def run_with_tool_calling_agent(task: str) -> str:
    """Let the ToolCallingAgent solve a task via structured JSON tool calls."""
    return str(tool_calling_agent.run(task))
