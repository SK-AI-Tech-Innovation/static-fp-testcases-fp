# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: Corrective RAG (CRAG) — grade retrieved docs, fall back to web search on "incorrect"
# WHY-CORRECT: CRAG runs a lightweight retrieval evaluator over each retrieved document, labeling
#   it correct / ambiguous / incorrect. If retrieval is poor, it triggers a corrective action
#   (web search / query rewrite) instead of feeding bad context to the generator. Discarding
#   low-grade documents and supplementing with web search is the intended CRAG correction step,
#   not dropped context.
# EXPECTED-WRONG: engine may flag "discarding retrieved documents loses context", claim the web
#   fallback bypasses the vector store incorrectly, or that all retrieved docs must be used.
# CORRECT-VERDICT: no findings
"""Corrective RAG: grade retrieved docs and fall back to web search when relevance is low."""
import anthropic

llm = anthropic.Anthropic()


def grade_doc(query: str, doc: str) -> str:
    resp = llm.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": f"Query: {query}\nDoc: {doc}\n"
                "Is this doc relevant? Answer exactly 'correct', 'ambiguous', or 'incorrect'.",
            }
        ],
    )
    return resp.content[0].text.strip().lower()


def web_search(query: str) -> list[str]:
    return [f"web result for {query}"]


def corrective_retrieve(query: str, retrieved: list[str]) -> list[str]:
    grades = {doc: grade_doc(query, doc) for doc in retrieved}
    kept = [d for d, g in grades.items() if g != "incorrect"]
    if not kept or all(g == "ambiguous" for g in grades.values()):
        kept = kept + web_search(query)
    return kept


if __name__ == "__main__":
    print(corrective_retrieve("capital of France", ["Paris is the capital of France."]))
