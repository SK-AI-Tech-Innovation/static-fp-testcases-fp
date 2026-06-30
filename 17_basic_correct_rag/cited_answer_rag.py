# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: ChromaDB (`chromadb`) + OpenAI chat with grounded, cited answers
# WHY-CORRECT: complete grounded RAG — retrieve top-k passages with stable source ids, number
#              them in the context, instruct the model to cite sources and to say "I don't know"
#              when the context lacks the answer. Grounding + citation + abstention are all present.
# EXPECTED-WRONG: engine invents "add a grounding/anti-hallucination guard" (already present)
# CORRECT-VERDICT: no findings
"""Answer with inline source citations, instructing the model to abstain when unsupported."""
import chromadb
from openai import OpenAI

llm = OpenAI()
client = chromadb.Client()
collection = client.get_or_create_collection(name="kb")


def index(documents: dict[str, str]) -> None:
    collection.add(ids=list(documents.keys()), documents=list(documents.values()))


def answer(question: str) -> str:
    results = collection.query(query_texts=[question], n_results=3)
    ids = results["ids"][0]
    docs = results["documents"][0]
    context = "\n".join(f"[{source}] {text}" for source, text in zip(ids, docs))
    system = (
        "Answer the question using only the numbered sources. Cite the source id in brackets "
        "after each claim. If the sources do not contain the answer, reply exactly: I don't know."
    )
    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    index({"src-1": "The Amazon River is the largest river by discharge.", "src-2": "Jupiter is the largest planet."})
    print(answer("Which is the largest river by discharge?"))
