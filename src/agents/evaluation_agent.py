from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.base import AgentResult, BaseAgent
from src.config import AppConfig
from src.utils.optional import optional_import


class EvaluationAgent(BaseAgent):
    name = "evaluation"

    def __init__(self, config: AppConfig):
        self.config = config
        self._sklearn, _ = optional_import("sklearn")

    def _metrics(self, labels: List[int], scores: List[float]) -> Dict[str, Any]:
        y_true = np.array(labels)
        y_score = np.array(scores)
        y_pred = (y_score >= self.config.model.fraud_threshold).astype(int)

        metrics: Dict[str, Any] = {}
        if self._sklearn is not None:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
                confusion_matrix,
            )  # type: ignore
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            try:
                metrics["auc"] = float(roc_auc_score(y_true, y_score))
            except Exception:
                metrics["auc"] = None
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        else:
            # Fallback metrics
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            metrics.update({
                "accuracy": float((tp + tn) / max(1, len(y_true))),
                "precision": float(tp / max(1, tp + fp)),
                "recall": float(tp / max(1, tp + fn)),
                "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
                "auc": None,
                "confusion_matrix": [[tn, fp], [fn, tp]],
            })
        return metrics

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        labels = payload.get("labels")
        scores = payload.get("scores")

        if labels is None or any(label is None for label in labels):
            return AgentResult(name=self.name, outputs={"metrics": None, "note": "Labels not available"})

        metrics = self._metrics(labels, scores)
        return AgentResult(name=self.name, outputs={"metrics": metrics})
