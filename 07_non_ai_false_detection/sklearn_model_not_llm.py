# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: classic scikit-learn pipeline (no LLM anywhere)
# WHY-CORRECT: "model", "predict", "embedding" here are CLASSICAL ML — no LLM/agent/RAG usage at all.
#              Static scope is LLM/AI-application patterns; this file has none.
# EXPECTED-WRONG: keyword-driven misread ("model", "embedding", "prompt"-like names) -> false AI-pattern
#                 detection -> spurious "best practice" findings about LLM usage that doesn't exist
# CORRECT-VERDICT: no findings ("LLM/AI 패턴이 발견되지 않았습니다")
"""Churn prediction with a scikit-learn gradient boosting model. No LLM involved."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class ChurnModel:
    """Trains and serves a tabular churn classifier."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(n_estimators=200, max_depth=3)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        # "embedding" below is a PCA-style feature embedding, not a vector DB / LLM embedding.
        embedding = self.scaler.transform(features)
        return self.model.predict_proba(embedding)[:, 1]


def build_user_prompt_string(username: str) -> str:
    # "prompt" here is a CLI shell prompt string — nothing to do with LLM prompting.
    return f"{username}@churn-cli$ "
