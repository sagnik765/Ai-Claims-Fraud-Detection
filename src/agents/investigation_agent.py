from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base import AgentResult, BaseAgent
from src.config import AppConfig
from src.utils.explanations import genai_rationale
from src.utils.openai_rationale import OpenAIRationaleGenerator


class InvestigationAgent(BaseAgent):
    name = "investigation"

    def __init__(self, config: AppConfig):
        self.config = config
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
        amount_stats = payload.get("amount_stats")
        results = []
        remaining = int(self.config.agents.genai_max_claims)

        for claim_id, record, score in zip(ids, records, scores):
            flags = self._build_flags(record, float(score))
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
                "flags": flags,
                "recommended_actions": self._recommended_actions(flags),
                "genai_rationale": genai["summary"],
                "decline_risk_reasons": genai["decline_risk_reasons"],
                "amount_rationale": genai["amount_rationale"],
                "decision_support": genai["decision_support"],
                "genai_disclaimer": genai["disclaimer"],
                "genai_mode": genai["genai_mode"],
            })

        return AgentResult(name=self.name, outputs={"investigations": results})
