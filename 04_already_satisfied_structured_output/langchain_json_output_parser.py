# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: LangChain `PydanticOutputParser` with format instructions + `.parse()` (and json_schema method)
# WHY-CORRECT: the parser injects schema-derived `format_instructions` into the prompt and `.parse()` validates
#              the model text against the Pydantic schema, raising on mismatch. The schema is the contract;
#              the consumer never hand-rolls regex/json.loads. This is canonical LangChain structured output.
# EXPECTED-WRONG: engine sees no `response_format=`/`with_structured_output(...)` token and flags it as
#                 "free-text parsing / not structured", missing that the OutputParser enforces the schema.
# CORRECT-VERDICT: no findings
"""Parse a model answer into a typed schema using LangChain's PydanticOutputParser."""
from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class BookInfo(BaseModel):
    title: str = Field(description="Title of the book")
    author: str = Field(description="Author's full name")
    year: int = Field(description="Year of publication")


_parser = PydanticOutputParser(pydantic_object=BookInfo)
_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using exactly this format:\n{format_instructions}"),
        ("human", "{query}"),
    ]
).partial(format_instructions=_parser.get_format_instructions())

# json_schema-equivalent enforcement: prompt carries the schema, parser validates the result.
_chain = _prompt | ChatOpenAI(model="gpt-4.1-mini", temperature=0) | _parser


def lookup_book(query: str) -> BookInfo:
    # _chain returns a validated BookInfo; the parser raises on any schema mismatch.
    return _chain.invoke({"query": query})
