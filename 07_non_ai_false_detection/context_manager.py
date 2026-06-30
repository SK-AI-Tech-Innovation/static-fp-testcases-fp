# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: Python contextlib resource scopes (@contextmanager + a class with __enter__/__exit__)
# WHY-CORRECT: "context" here is the Python context-manager protocol used for deterministic
#              resource cleanup (a temp dir, a timer). It is not an LLM context window, not
#              prompt context, and there is no model invocation of any kind.
# EXPECTED-WRONG: keyword "context" (context_manager, with_context) -> false "LLM context
#                 window" detection -> spurious "context length exceeded / no max context"
#                 finding.
# CORRECT-VERDICT: no findings
"""Python resource context managers (temp dir, timer). Not an LLM context window."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def temporary_directory(prefix: str = "scratch-") -> Iterator[str]:
    """Create a temp directory and remove it on scope exit, even on error."""
    path = tempfile.mkdtemp(prefix=prefix)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class Timer:
    """Context manager that measures wall-clock duration of a `with` block."""

    def __init__(self, label: str = "block") -> None:
        self.label = label
        self.elapsed_s: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.elapsed_s = time.perf_counter() - self._start
        return False


def write_in_context(filename: str, data: str) -> int:
    """Write a file inside a temporary-directory context and return its byte size."""
    with temporary_directory() as workdir:
        full = os.path.join(workdir, filename)
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(data)
        return os.path.getsize(full)
