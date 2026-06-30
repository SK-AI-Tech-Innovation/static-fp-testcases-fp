# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangChain 1.0 — legacy chains moved to `langchain_classic.chains`
# WHY-CORRECT: In LangChain 1.0, the legacy chain abstractions (LLMChain, RetrievalQA, etc.)
#   were intentionally relocated to the `langchain_classic` package to keep the core lean.
#   Importing `RetrievalQA` and `LLMChain` from `langchain_classic.chains` is the correct,
#   supported path for code that still uses these classic constructs.
# EXPECTED-WRONG: engine may flag `langchain_classic` as a non-existent module, or push the old
#   `from langchain.chains import LLMChain` path that no longer resolves in 1.0.
# CORRECT-VERDICT: no findings
"""Use classic LLMChain/RetrievalQA via the langchain_classic package in LangChain 1.0."""
from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def build_llm_chain() -> LLMChain:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template("Summarize: {text}")
    return LLMChain(llm=llm, prompt=prompt)


def build_qa(retriever) -> RetrievalQA:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


if __name__ == "__main__":
    chain = build_llm_chain()
    print(type(chain).__name__)
