# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: Microsoft Semantic Kernel: Kernel + chat service + @kernel_function plugin
# WHY-CORRECT: This is idiomatic Semantic Kernel: a Kernel with an added chat-completion
#              service, a native plugin class whose methods are decorated with
#              @kernel_function (with name/description), and invocation via
#              kernel.invoke(...). Prompt construction, function-calling, and the model call
#              are owned by the kernel; there is no raw OpenAI client to audit.
# EXPECTED-WRONG: @kernel_function methods with no visible prompt may trip "tool without
#                 description" (descriptions live in the decorator) and the implicit model
#                 call may trigger "no model invocation found".
# CORRECT-VERDICT: no findings
"""Idiomatic Microsoft Semantic Kernel: kernel, chat service, @kernel_function plugin."""
from __future__ import annotations

import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function


class TimePlugin:
    """A native Semantic Kernel plugin exposing deterministic time helpers."""

    @kernel_function(name="today", description="Return today's date as ISO-8601.")
    def today(self) -> str:
        from datetime import date

        return date.today().isoformat()

    @kernel_function(name="days_between", description="Whole days between two ISO dates.")
    def days_between(self, start: str, end: str) -> int:
        from datetime import date

        return (date.fromisoformat(end) - date.fromisoformat(start)).days


def build_kernel() -> Kernel:
    """Create a kernel with an OpenAI chat service and the time plugin registered."""
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4o"))
    kernel.add_plugin(TimePlugin(), plugin_name="time")
    return kernel


async def ask(kernel: Kernel, question: str) -> str:
    """Invoke a templated prompt function that may call the time plugin."""
    prompt = "{{$input}}\n\nUse the time plugin when a date is needed."
    func = kernel.add_function(
        plugin_name="qa",
        function_name="answer",
        prompt=prompt,
    )
    result = await kernel.invoke(func, input=question)
    return str(result)


if __name__ == "__main__":
    asyncio.run(ask(build_kernel(), "How many days until the new year?"))
