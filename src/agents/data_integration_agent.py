from __future__ import annotations

from typing import Any, Dict

from src.agents.base import AgentResult, BaseAgent
from src.config import AppConfig
from src.pipelines.data_integration import extract_modalities, load_records


class DataIntegrationAgent(BaseAgent):
    name = "data_integration"

    def __init__(self, config: AppConfig, prefer_spark: bool = False):
        self.config = config
        self.prefer_spark = prefer_spark

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        data_path = payload["data_path"]
        records = load_records(data_path, prefer_spark=self.prefer_spark)
        modalities = extract_modalities(records, self.config)
        return AgentResult(
            name=self.name,
            outputs={
                "records": modalities["records"],
                "structured_records": modalities["structured_records"],
                "ids": modalities["ids"],
                "texts": modalities["texts"],
                "images": modalities["images"],
                "labels": modalities["labels"],
            },
        )
