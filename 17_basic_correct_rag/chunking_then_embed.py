# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: LangChain RecursiveCharacterTextSplitter + Chroma + OpenAI
# WHY-CORRECT: correct ingestion pipeline — split long text into overlapping chunks (chunk_size
#              with chunk_overlap set), embed and store each chunk, then retrieve and answer.
#              Overlap is already configured, so the common "add chunk overlap" advice is moot.
# EXPECTED-WRONG: engine invents "add chunk overlap" (already present) or "use semantic chunking"
# CORRECT-VERDICT: no findings
"""Split a document into overlapping chunks, embed/store them, and answer over retrieval."""
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
llm = ChatOpenAI(model="gpt-4.1-mini")


def build_store(text: str) -> Chroma:
    chunks = splitter.split_text(text)
    return Chroma.from_texts(chunks, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))


def answer(question: str, text: str) -> str:
    store = build_store(text)
    docs = store.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content


if __name__ == "__main__":
    document = "The mitochondrion is the powerhouse of the cell. " * 40
    print(answer("What is the powerhouse of the cell?", document))
