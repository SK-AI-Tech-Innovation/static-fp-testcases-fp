# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: langchain v0.0.x (pre-LCEL) with openai-python v0.x — authentic legacy API
# WHY-CORRECT: Before LangChain Expression Language (LCEL), the canonical pattern was
#   LLMChain(llm=OpenAI(), prompt=PromptTemplate(...)) with top-level `from langchain import ...`
#   imports and chain.run(...). This was correct and idiomatic for that LangChain era.
# EXPECTED-WRONG: engine may flag LLMChain / `from langchain import` as deprecated, push LCEL
#   (prompt | llm | parser) or langchain_openai imports, or call chain.run a bug.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy LangChain (pre-LCEL) LLMChain over an OpenAI LLM."""
from langchain import LLMChain, OpenAI, PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(
    llm=OpenAI(temperature=0.7, model_name="text-davinci-003"),
    prompt=prompt,
)


def name_company(product: str) -> str:
    return chain.run(product).strip()


if __name__ == "__main__":
    print(name_company("colorful socks"))
