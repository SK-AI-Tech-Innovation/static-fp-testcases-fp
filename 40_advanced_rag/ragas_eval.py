# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: RAGAS — reference-free RAG evaluation (faithfulness, answer/context relevancy)
# WHY-CORRECT: RAGAS evaluates a RAG pipeline with LLM-as-judge metrics. Faithfulness and
#   ContextRelevance are largely reference-free (no ground-truth answer needed); they score how
#   grounded the answer is in retrieved context and how relevant the context is to the question.
#   Building an EvaluationDataset of samples and calling `evaluate(dataset, metrics=[...])` is the
#   documented API; the absence of a gold answer for these metrics is by design.
# EXPECTED-WRONG: engine may flag "evaluation without ground-truth labels", claim faithfulness
#   needs a reference answer, or that an LLM judging its own pipeline is circular/invalid.
# CORRECT-VERDICT: no findings
"""Evaluate a RAG pipeline with RAGAS reference-free metrics (faithfulness, context relevance)."""
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRelevance, Faithfulness
from langchain_openai import ChatOpenAI


def build_dataset() -> EvaluationDataset:
    samples = [
        {
            "user_input": "What is the capital of France?",
            "retrieved_contexts": ["Paris is the capital of France."],
            "response": "The capital of France is Paris.",
        }
    ]
    return EvaluationDataset.from_list(samples)


def run_eval():
    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    dataset = build_dataset()
    return evaluate(
        dataset=dataset,
        metrics=[Faithfulness(llm=judge), ContextRelevance(llm=judge)],
    )


if __name__ == "__main__":
    print(run_eval())
