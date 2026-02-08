from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.agents.base import AgentResult, BaseAgent
from src.config import AppConfig
from src.utils.explanations import genai_rationale
from src.utils.openai_rationale import OpenAIRationaleGenerator
from src.utils.optional import optional_import


class EvaluationAgent(BaseAgent):
    name = "evaluation"

    def __init__(self, config: AppConfig):
        self.config = config
        self._sklearn, _ = optional_import("sklearn")
        self._openai = OpenAIRationaleGenerator(config.agents.genai_model)

    def _eligible_for_llm(self, score: float, remaining: int) -> bool:
        if not self.config.agents.genai_enabled:
            return False
        if self.config.agents.genai_provider != "openai":
            return False
        if not self._openai.available():
            return False
        if remaining <= 0:
            return False
        if self.config.agents.genai_scope == "all":
            return True
        return score >= self.config.agents.genai_min_score

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
        ids = payload.get("ids")
        records = payload.get("records")
        amount_stats = payload.get("amount_stats")

        if labels is None or any(label is None for label in labels):
            outputs: Dict[str, Any] = {"metrics": None, "note": "Labels not available"}
            if ids and records and scores:
                outputs["claim_evaluations"] = self._claim_rationales(ids, records, scores, amount_stats)
            return AgentResult(name=self.name, outputs=outputs)

        metrics = self._metrics(labels, scores)
        outputs: Dict[str, Any] = {"metrics": metrics}
        if ids and records and scores:
            outputs["claim_evaluations"] = self._claim_rationales(ids, records, scores, amount_stats)
        return AgentResult(name=self.name, outputs=outputs)

    def _claim_rationales(
        self,
        ids: List[str],
        records: List[Dict[str, Any]],
        scores: List[float],
        amount_stats: Any,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        remaining = int(self.config.agents.genai_max_claims)
        for claim_id, record, score in zip(ids, records, scores):
            genai = genai_rationale(
                record=record,
                score=float(score),
                threshold=self.config.model.fraud_threshold,
                stats=amount_stats,
                disclaimer=self.config.agents.genai_disclaimer,
            )
            if self._eligible_for_llm(float(score), remaining):
                summary = self._openai.summarize({
                    "score": float(score),
                    "threshold": self.config.model.fraud_threshold,
                    "decline_risk_reasons": genai["decline_risk_reasons"],
                    "amount_rationale": genai["amount_rationale"],
                    "decision_support": genai["decision_support"],
                })
                if summary:
                    genai["summary"] = summary
                    genai["genai_mode"] = "openai"
                    remaining -= 1
            results.append({
                "claim_id": claim_id,
                "score": float(score),
                "genai_rationale": genai["summary"],
                "decline_risk_reasons": genai["decline_risk_reasons"],
                "amount_rationale": genai["amount_rationale"],
                "decision_support": genai["decision_support"],
                "genai_disclaimer": genai["disclaimer"],
                "genai_mode": genai["genai_mode"],
            })
        return results
