from __future__ import annotations

from typing import Any, Dict

from src.agents.data_integration_agent import DataIntegrationAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.investigation_agent import InvestigationAgent
from src.agents.subrogation_agent import SubrogationSalvageAgent
from src.config import AppConfig
from src.models.multimodal import MultimodalFraudModel
from src.utils.explanations import compute_amount_stats


class FraudOrchestrator:
    def __init__(self, config: AppConfig, prefer_spark: bool = False):
        self.config = config
        self.prefer_spark = prefer_spark
        self.model = MultimodalFraudModel(
            text_vectorizer=config.model.text_vectorizer,
            text_max_features=config.model.max_text_features,
            image_feature_dim=config.model.image_feature_dim,
            structured_hash_features=config.model.structured_hash_features,
            model_type=config.model.model_type,
        )

        self.data_agent = DataIntegrationAgent(config, prefer_spark=prefer_spark)
        self.eval_agent = EvaluationAgent(config)
        self.investigation_agent = InvestigationAgent(config)
        self.subrogation_agent = SubrogationSalvageAgent(config)

    def run(self, mode: str, data_path: str) -> Dict[str, Any]:
        data_payload = self.data_agent.run({"data_path": data_path}).outputs
        amount_stats = compute_amount_stats(data_payload["records"])

        if mode == "train":
            labels = [label for label in data_payload["labels"] if label is not None]
            if not labels:
                raise ValueError("Training requires labels in the dataset")
            self.model.train(
                texts=data_payload["texts"],
                image_paths=data_payload["images"],
                structured_records=data_payload["structured_records"],
                labels=data_payload["labels"],
            )

        scores = self.model.predict_proba(
            texts=data_payload["texts"],
            image_paths=data_payload["images"],
            structured_records=data_payload["structured_records"],
        )

        eval_result = self.eval_agent.run({
            "labels": data_payload["labels"],
            "scores": scores.tolist(),
            "ids": data_payload["ids"],
            "records": data_payload["records"],
            "amount_stats": amount_stats,
        }).outputs

        investigation_result = self.investigation_agent.run({
            "ids": data_payload["ids"],
            "records": data_payload["records"],
            "scores": scores.tolist(),
            "amount_stats": amount_stats,
        }).outputs

        subrogation_result = self.subrogation_agent.run({
            "ids": data_payload["ids"],
            "records": data_payload["records"],
            "texts": data_payload["texts"],
        }).outputs

        return {
            "scores": scores.tolist(),
            "evaluation": eval_result,
            "investigation": investigation_result,
            "subrogation": subrogation_result,
        }
