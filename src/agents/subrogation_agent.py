from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base import AgentResult, BaseAgent
from src.config import AppConfig
from src.utils.text import normalize_text


class SubrogationSalvageAgent(BaseAgent):
    name = "subrogation_salvage"

    def __init__(self, config: AppConfig):
        self.config = config

    def _score_keywords(self, text: str) -> float:
        text = normalize_text(text)
        hits = 0
        for kw in self.config.agents.subrogation_keywords:
            if kw in text:
                hits += 1
        return hits / max(1, len(self.config.agents.subrogation_keywords))

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        texts = payload["texts"]
        records = payload["records"]
        ids = payload["ids"]
        results: List[Dict[str, Any]] = []

        for claim_id, text, record in zip(ids, texts, records):
            kw_score = self._score_keywords(text)
            salvage_flag = False
            if float(record.get("total_loss", 0) or 0) > 0:
                salvage_flag = True
            if "salvage" in normalize_text(text):
                salvage_flag = True

            results.append({
                "claim_id": claim_id,
                "subrogation_likelihood": float(kw_score),
                "salvage_indicated": bool(salvage_flag),
                "notes": "Keyword-based baseline, extend with liability signals",
            })

        return AgentResult(name=self.name, outputs={"subrogation": results})
