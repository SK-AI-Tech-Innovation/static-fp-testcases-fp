# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: LangChain Expression Language (LCEL): prompt | model | output_parser chain
# WHY-CORRECT: This is the canonical LCEL idiom: a ChatPromptTemplate piped into a chat
#              model piped into a StrOutputParser using the `|` operator, producing a
#              Runnable invoked with .invoke(...). The chain composition, prompt formatting,
#              model call, and output parsing are all handled by LangChain Runnables; there
#              is no raw client call or manual response[...] indexing to audit.
# EXPECTED-WRONG: the overloaded `|` operator may be misread as a bitwise op or unusual
#                 control flow; static checks may flag "no explicit model.generate / no
#                 response parsing" since parsing is the piped StrOutputParser.
# CORRECT-VERDICT: no findings
"""Idiomatic LangChain LCEL pipe chain: prompt | model | parser as a Runnable."""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You translate text into {language}. Reply with only the translation."),
        ("human", "{text}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# The `|` operator composes Runnables left-to-right into a single chain.
translation_chain = prompt | model | StrOutputParser()


def translate(text: str, language: str = "French") -> str:
    """Run the LCEL chain to translate a piece of text."""
    return translation_chain.invoke({"text": text, "language": language})


def translate_batch(texts: list[str], language: str = "French") -> list[str]:
    """Translate many inputs using the chain's built-in batch support."""
    return translation_chain.batch([{"text": t, "language": language} for t in texts])
