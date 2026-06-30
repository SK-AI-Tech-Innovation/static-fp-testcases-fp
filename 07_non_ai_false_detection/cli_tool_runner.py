# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: A CLI build-tool subcommand dispatcher (registry of "tools" like make/just/npm)
# WHY-CORRECT: "tool" here is a registered CLI subcommand (build/lint/test) dispatched by
#              name from argv. There is no LLM tool-calling, no function/tool schema, no
#              agent that selects tools, and no model. It is a plain command runner.
# EXPECTED-WRONG: keyword "tool" / "register_tool" / "ToolRegistry" -> false "LLM tool-use
#                 pattern" detection -> spurious "tool has no description / no JSON schema"
#                 finding.
# CORRECT-VERDICT: no findings
"""CLI build-tool dispatcher: named subcommands run shell steps. Not LLM tool-calling."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    """A named CLI subcommand mapped to a shell command sequence."""

    name: str
    description: str
    run: Callable[[], int]


@dataclass
class ToolRegistry:
    """Registry that maps subcommand names to build tools."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register_tool(self, tool: Tool) -> None:
        """Add a tool to the registry, rejecting duplicate names."""
        if tool.name in self.tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self.tools[tool.name] = tool

    def dispatch(self, name: str) -> int:
        """Run the named tool and return its process exit code."""
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        return self.tools[name].run()


def _shell(command: list[str]) -> int:
    """Run a shell command and return its exit code."""
    return subprocess.run(command, check=False).returncode


def build_default_registry() -> ToolRegistry:
    """Construct the standard build/lint/test tools for this project."""
    registry = ToolRegistry()
    registry.register_tool(Tool("build", "compile sources", lambda: _shell(["make", "build"])))
    registry.register_tool(Tool("lint", "run the linter", lambda: _shell(["ruff", "check", "."])))
    registry.register_tool(Tool("test", "run unit tests", lambda: _shell(["pytest", "-q"])))
    return registry


def main(argv: list[str]) -> int:
    """Entry point: dispatch the first CLI argument to a registered tool."""
    if not argv:
        print("usage: runner <tool>", file=sys.stderr)
        return 2
    return build_default_registry().dispatch(argv[0])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
