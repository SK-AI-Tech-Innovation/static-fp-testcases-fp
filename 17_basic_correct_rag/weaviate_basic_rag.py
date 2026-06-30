# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: weaviate-client v4 (`weaviate`) near_text query + OpenAI chat
# WHY-CORRECT: current weaviate-client v4 API — connect_to_local, get the collection, insert
#              objects, retrieve with collection.query.near_text(limit=k), read properties from
#              objects, inject into the prompt, answer. Connection is closed in a finally block.
# EXPECTED-WRONG: engine invents "use hybrid/near_vector", "add reranking", or "tune alpha"
# CORRECT-VERDICT: no findings
"""Query a Weaviate collection with near_text and answer over the retrieved objects."""
import weaviate
from openai import OpenAI

llm = OpenAI()


def answer(question: str, documents: list[str]) -> str:
    client = weaviate.connect_to_local()
    try:
        docs = client.collections.get("Doc")
        with docs.batch.dynamic() as batch:
            for doc in documents:
                batch.add_object(properties={"text": doc})
        result = docs.query.near_text(query=question, limit=3)
        context = "\n\n".join(obj.properties["text"] for obj in result.objects)
    finally:
        client.close()
    prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(answer("What is Python?", ["Python is a high-level programming language."]))
