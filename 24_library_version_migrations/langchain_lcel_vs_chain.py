# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: LangChain — legacy LLMChain vs. modern LCEL (`prompt | model | parser`)
# WHY-CORRECT: This file intentionally shows both styles as a side-by-side migration. The
#   legacy LLMChain path is correct on the older API; the LCEL pipe composition is the
#   idiomatic v0.2/v0.3 replacement. Both functions are valid in their own version context;
#   keeping both is normal during an incremental migration.
# EXPECTED-WRONG: engine may flag LLMChain as deprecated and demand it be deleted in favor of
#   LCEL, or flag the `prompt | model` pipe operator as a misuse / suspicious operator overload.
# CORRECT-VERDICT: no findings
"""Side-by-side legacy LLMChain and modern LCEL pipe composition for the same task."""
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

PROMPT = PromptTemplate.from_template("Translate to French: {text}")
MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def translate_legacy(text: str) -> str:
    """Legacy v0.1-style chain object."""
    chain = LLMChain(llm=MODEL, prompt=PROMPT)
    return chain.run(text=text)


def translate_lcel(text: str) -> str:
    """Modern LCEL: prompt | model | parser."""
    chain = PROMPT | MODEL | StrOutputParser()
    return chain.invoke({"text": text})


if __name__ == "__main__":
    print(translate_lcel("Good morning"))
