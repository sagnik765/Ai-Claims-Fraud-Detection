from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base import AgentResult, BaseAgent
from src.config import AppConfig


class InvestigationAgent(BaseAgent):
    name = "investigation"

    def __init__(self, config: AppConfig):
        self.config = config

    def _build_flags(self, record: Dict[str, Any], score: float) -> List[str]:
        flags: List[str] = []
        if score >= self.config.model.fraud_threshold:
            flags.append("high_fraud_score")
        if float(record.get("late_reported", 0) or 0) > 0:
            flags.append("late_reported_loss")
        if float(record.get("policy_age_days", 0) or 0) < 30:
            flags.append("policy_recently_bound")
        if float(record.get("prior_claims_count", 0) or 0) >= 2:
            flags.append("multiple_claims_same_asset")
        if float(record.get("multiple_parties", 0) or 0) > 0:
            flags.append("multiple_parties_involved")
        if float(record.get("injury_reported", 0) or 0) > 0:
            flags.append("injury_reported")
        return flags

    def _recommended_actions(self, flags: List[str]) -> List[str]:
        actions = [
            "Verify policy status and coverage at loss date",
            "Request recorded statement and supporting documentation",
            "Cross-check claim history and prior losses",
        ]
        if "late_reported_loss" in flags:
            actions.append("Validate loss timeline with third-party data sources")
        if "multiple_parties_involved" in flags:
            actions.append("Collect independent witness or police reports")
        if "injury_reported" in flags:
            actions.append("Review medical bills for consistency and treatment patterns")
        if "multiple_claims_same_asset" in flags:
            actions.append("Inspect asset for pre-existing damage or prior repairs")
        return actions

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        ids = payload["ids"]
        records = payload["records"]
        scores = payload["scores"]
        results = []

        for claim_id, record, score in zip(ids, records, scores):
            flags = self._build_flags(record, float(score))
            results.append({
                "claim_id": claim_id,
                "score": float(score),
                "flags": flags,
                "recommended_actions": self._recommended_actions(flags),
            })

        return AgentResult(name=self.name, outputs={"investigations": results})
