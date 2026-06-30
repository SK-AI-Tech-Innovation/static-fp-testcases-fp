# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: DSPy Signature + dspy.Module composing ChainOfThought (no hand-written prompts)
# WHY-CORRECT: DSPy's whole point is that you declare a typed Signature (inputs -> outputs)
#              and compose modules like ChainOfThought; DSPy compiles the actual prompt for
#              you. The ABSENCE of a literal prompt string is correct and idiomatic, not a
#              bug. dspy.configure(lm=...) sets the backing model.
# EXPECTED-WRONG: a keyword scanner that expects an explicit prompt/template may flag "no
#                 system prompt / prompt not found" or "instructions missing", or read the
#                 docstring-as-instruction Signature as an undefined prompt -> spurious
#                 "prompt management" finding. ChainOfThought is also not a LangChain chain.
# CORRECT-VERDICT: no findings
"""Idiomatic DSPy module: a declarative Signature compiled via ChainOfThought."""
from __future__ import annotations

import dspy


class ClassifySentiment(dspy.Signature):
    """Classify the sentiment of a customer review."""

    review: str = dspy.InputField(desc="raw customer review text")
    sentiment: str = dspy.OutputField(desc="one of: positive, neutral, negative")
    confidence: float = dspy.OutputField(desc="model confidence between 0 and 1")


class SentimentPipeline(dspy.Module):
    """Composes reasoning over the ClassifySentiment signature.

    No prompt string is written by hand: DSPy compiles the prompt from the
    Signature and the chosen reasoning strategy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(ClassifySentiment)

    def forward(self, review: str) -> dspy.Prediction:
        return self.classify(review=review)


def configure_lm(model: str = "openai/gpt-4o-mini") -> None:
    """Set the backing language model for all DSPy modules in this process."""
    dspy.configure(lm=dspy.LM(model))


def classify(review: str) -> tuple[str, float]:
    pipeline = SentimentPipeline()
    prediction = pipeline(review=review)
    return prediction.sentiment, float(prediction.confidence)
