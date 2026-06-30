// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: LlamaIndex.TS — `VectorStoreIndex.fromDocuments(...).asQueryEngine().query(...)`
// WHY-CORRECT: this is the canonical LlamaIndex.TS RAG path: build an index from documents, derive a query
//              engine, and query it. Retrieval grounding is handled by the framework; the code is complete.
// EXPECTED-WRONG: a Python-centric engine may not map LlamaIndex.TS to its retrieval patterns and flag
//                 "no retrieval / prompt not grounded", "missing system prompt", or mis-detect it as non-AI.
// CORRECT-VERDICT: no findings
/** Minimal LlamaIndex.TS RAG query engine over a couple of in-memory documents. */
import { Document, VectorStoreIndex } from "llamaindex";

const docs = [
  new Document({ text: "ACME returns digital goods within 14 days if unused." }),
  new Document({ text: "ACME refunds go to the original payment method in 5 business days." }),
];

export async function answerFromDocs(question: string): Promise<string> {
  const index = await VectorStoreIndex.fromDocuments(docs);
  const queryEngine = index.asQueryEngine();
  const { response } = await queryEngine.query({ query: question });
  // `response` is grounded in the retrieved document chunks by the engine.
  return response;
}
