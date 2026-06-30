# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: LangChain MultiQueryRetriever — LLM expands one query into several, union the results
# WHY-CORRECT: MultiQueryRetriever uses an LLM to generate several paraphrases of the user's
#   question, retrieves for each, and returns the UNION of the unique documents. This widens recall
#   for under-specified queries. Generating extra queries and merging results is the intended
#   behavior; the LLM-produced variants drive retrieval, not the answer.
# EXPECTED-WRONG: engine may flag "issuing multiple retrieval calls per query" as wasteful, claim
#   the generated sub-queries are hallucinated input, or that only the original query should be used.
# CORRECT-VERDICT: no findings
"""Expand a query with MultiQueryRetriever and union retrieved documents in LangChain 1.0."""
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def build_retriever(texts: list[str]):
    store = InMemoryVectorStore.from_texts(texts, embedding=OpenAIEmbeddings())
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return MultiQueryRetriever.from_llm(retriever=store.as_retriever(), llm=llm)


def search(query: str, texts: list[str]):
    retriever = build_retriever(texts)
    return retriever.invoke(query)


if __name__ == "__main__":
    docs = ["Paris is the capital of France.", "France uses the euro."]
    print([d.page_content for d in search("Tell me about France", docs)])
