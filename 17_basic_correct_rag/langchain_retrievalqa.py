# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: LangChain (`langchain`, `langchain_openai`, `langchain_chroma`) LCEL retrieval chain
# WHY-CORRECT: idiomatic modern LangChain RAG — build a Chroma vector store from texts with
#              OpenAIEmbeddings, expose a retriever, compose create_stuff_documents_chain with a
#              ChatPromptTemplate (context + input), and wrap with create_retrieval_chain.
# EXPECTED-WRONG: engine invents "use deprecated RetrievalQA", "add reranking", or "tune k"
# CORRECT-VERDICT: no findings
"""Answer questions with a LangChain LCEL retrieval chain over a Chroma store."""
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the context:\n\n{context}"),
        ("human", "{input}"),
    ]
)


def build_chain(documents: list[str]):
    store = Chroma.from_texts(documents, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = store.as_retriever(search_kwargs={"k": 3})
    combine = create_stuff_documents_chain(ChatOpenAI(model="gpt-4.1-mini"), prompt)
    return create_retrieval_chain(retriever, combine)


def answer(question: str, documents: list[str]) -> str:
    chain = build_chain(documents)
    return chain.invoke({"input": question})["answer"]


if __name__ == "__main__":
    print(answer("Who wrote Hamlet?", ["Hamlet was written by William Shakespeare."]))
