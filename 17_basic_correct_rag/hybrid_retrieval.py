# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: LangChain EnsembleRetriever (BM25 + Chroma dense) + OpenAI
# WHY-CORRECT: idiomatic hybrid RAG — combine a BM25Retriever (sparse keyword) and a Chroma
#              dense retriever via EnsembleRetriever with weights, retrieve the fused results,
#              inject into the prompt, answer. Dense + keyword hybrid is already implemented.
# EXPECTED-WRONG: engine invents "add hybrid/keyword search" (already present) or "add reranking"
# CORRECT-VERDICT: no findings
"""Hybrid dense+keyword retrieval with an ensemble retriever, then answer the question."""
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4.1-mini")


def build_retriever(documents: list[str]) -> EnsembleRetriever:
    bm25 = BM25Retriever.from_texts(documents)
    bm25.k = 3
    dense = Chroma.from_texts(
        documents, embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    ).as_retriever(search_kwargs={"k": 3})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])


def answer(question: str, documents: list[str]) -> str:
    retriever = build_retriever(documents)
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content


if __name__ == "__main__":
    corpus = ["The Great Wall of China is visible from low orbit.", "Saturn has prominent rings."]
    print(answer("Which planet has rings?", corpus))
