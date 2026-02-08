from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import os

import yaml


@dataclass
class DataConfig:
    text_fields: List[str] = field(default_factory=lambda: [
        "claim_description",
        "loss_description",
        "adjuster_notes",
    ])
    image_field: str = "image_paths"
    label_field: str = "is_fraud"


@dataclass
class ModelConfig:
    text_vectorizer: str = "tfidf"  # tfidf or hashing
    max_text_features: int = 5000
    image_feature_dim: int = 64
    structured_hash_features: int = 512
    model_type: str = "random_forest"  # logreg or random_forest
    fraud_threshold: float = 0.6


@dataclass
class AgentConfig:
    investigation_rules: List[str] = field(default_factory=lambda: [
        "multiple_claims_same_asset",
        "late_reported_loss",
        "policy_recently_bound",
    ])
    subrogation_keywords: List[str] = field(default_factory=lambda: [
        "rear-ended",
        "hit and run",
        "other driver",
        "at fault",
        "third party",
        "salvage",
        "total loss",
    ])
    genai_disclaimer: str = (
        "Model-generated rationale for investigator review only; not an automated coverage decision."
    )


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)


def load_config(path: Optional[str] = None) -> AppConfig:
    if path is None:
        return AppConfig()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = DataConfig(**raw.get("data", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    agent_cfg = AgentConfig(**raw.get("agents", {}))
    return AppConfig(data=data_cfg, model=model_cfg, agents=agent_cfg)
