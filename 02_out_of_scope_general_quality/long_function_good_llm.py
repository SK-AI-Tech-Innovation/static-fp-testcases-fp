# ACE-FP-EXPECT: scoped-out
# CATEGORY: 02_out_of_scope_general_quality
# SOURCE: a single long function performing a correct RAG flow (retrieve -> assemble context -> answer)
# WHY-CORRECT: the RAG pattern is correct — context is grounded, the system prompt instructs the model to
#              answer ONLY from provided context and to say "I don't know" otherwise, and max_tokens is set.
#              The sole "defect" is that everything lives in one long function — a general refactoring/length
#              concern that is explicitly out-of-scope for the static LLM-pattern engine.
# EXPECTED-WRONG: "function too long / extract helpers / cyclomatic complexity" style refactoring findings.
# CORRECT-VERDICT: no findings (length/structure is out-of-scope; the RAG grounding is already correct)
"""End-to-end RAG answer in one long function (correct grounding; just not decomposed)."""
from __future__ import annotations

import math

import anthropic

client = anthropic.Anthropic()


def answer_question_with_rag(question: str, documents: list[dict]) -> str:
    """Retrieve top docs by cosine similarity then answer strictly from that context."""
    # --- 1. embed the query (toy embedding for the testcase) ---
    def embed(text: str) -> list[float]:
        vec = [0.0] * 32
        for i, ch in enumerate(text.lower()):
            vec[(ord(ch) + i) % 32] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    q_vec = embed(question)

    # --- 2. score every document by cosine similarity ---
    scored = []
    for doc in documents:
        d_vec = embed(doc["text"])
        dot = sum(a * b for a, b in zip(q_vec, d_vec))
        scored.append((dot, doc))

    # --- 3. take the top 3 chunks ---
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [doc for _, doc in scored[:3]]

    # --- 4. assemble grounded context ---
    context_blocks = []
    for idx, doc in enumerate(top, start=1):
        context_blocks.append(f"[{idx}] (source: {doc.get('source', 'unknown')})\n{doc['text']}")
    context = "\n\n".join(context_blocks)

    # --- 5. build the grounded prompt with an anti-hallucination instruction ---
    system = (
        "You are a question-answering assistant. Answer ONLY using the provided context. "
        "If the context does not contain the answer, reply exactly: I don't know. "
        "Cite the bracketed source numbers you used."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    # --- 6. call the model and return the text ---
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(parts).strip()


if __name__ == "__main__":
    docs = [
        {"text": "The Eiffel Tower is 330 metres tall.", "source": "wiki:eiffel"},
        {"text": "Paris is the capital of France.", "source": "wiki:paris"},
    ]
    print(answer_question_with_rag("How tall is the Eiffel Tower?", docs))
