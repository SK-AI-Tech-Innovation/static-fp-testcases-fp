# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: a toy blockchain ledger + a fluent method-chaining query builder
# WHY-CORRECT: "chain" here is (1) a cryptographic block chain (hash-linked blocks) and
#              (2) fluent method chaining on a builder object. Neither is a LangChain
#              Chain/Runnable; there is no LLM, prompt, or model anywhere.
# EXPECTED-WRONG: keyword "chain" (Blockchain, chain, .chain()) -> false "LangChain chain"
#                 detection -> spurious findings about LCEL / Runnable best practices.
# CORRECT-VERDICT: no findings
"""A hash-linked blockchain ledger and a fluent query builder. Not a LangChain chain."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class Block:
    """A single ledger block linked to its predecessor by hash."""

    index: int
    payload: dict[str, object]
    previous_hash: str
    timestamp: float = field(default_factory=time.time)

    def compute_hash(self) -> str:
        body = json.dumps(
            {
                "index": self.index,
                "payload": self.payload,
                "previous_hash": self.previous_hash,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(body).hexdigest()


class Blockchain:
    """An append-only chain of blocks validated by hash linkage."""

    def __init__(self) -> None:
        genesis = Block(index=0, payload={"genesis": True}, previous_hash="0" * 64)
        self.chain: list[Block] = [genesis]

    def add_block(self, payload: dict[str, object]) -> Block:
        previous = self.chain[-1]
        block = Block(
            index=previous.index + 1,
            payload=payload,
            previous_hash=previous.compute_hash(),
        )
        self.chain.append(block)
        return block

    def is_valid(self) -> bool:
        for prev, current in zip(self.chain, self.chain[1:]):
            if current.previous_hash != prev.compute_hash():
                return False
        return True


class QueryChain:
    """A fluent builder whose methods chain to assemble a SQL-like query."""

    def __init__(self, table: str) -> None:
        self._table = table
        self._filters: list[str] = []
        self._order: str | None = None

    def where(self, clause: str) -> "QueryChain":
        self._filters.append(clause)
        return self

    def order_by(self, column: str) -> "QueryChain":
        self._order = column
        return self

    def build(self) -> str:
        sql = f"SELECT * FROM {self._table}"
        if self._filters:
            sql += " WHERE " + " AND ".join(self._filters)
        if self._order:
            sql += f" ORDER BY {self._order}"
        return sql


def example() -> str:
    return QueryChain("orders").where("status = 'paid'").order_by("created_at").build()
