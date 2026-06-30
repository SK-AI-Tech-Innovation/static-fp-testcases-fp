# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: langchain (legacy LLMChain) + langchain-core LCEL (new pipe syntax) coexisting in one module
# WHY-CORRECT: a half-migrated module legitimately runs both styles. LLMChain still works (deprecated, not removed) and the LCEL `prompt | llm | parser` path also works. Both produce valid output during incremental migration.
# EXPECTED-WRONG: engine may flag LLMChain as "removed/broken" or claim mixing legacy chains with LCEL in one file is invalid.
# CORRECT-VERDICT: no findings
"""A module mid-migration: legacy LLMChain alongside new LCEL pipelines."""

from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate.from_template("Translate to French: {text}")


def legacy_translate(text: str) -> str:
    """Old code path still relying on LLMChain."""
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text=text)


def modern_translate(text: str) -> str:
    """New LCEL code path."""
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


if __name__ == "__main__":
    print(legacy_translate("good morning"))
    print(modern_translate("good night"))
