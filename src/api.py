from __future__ import annotations

from typing import Any, Dict, List
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import load_config
from src.orchestrator import FraudOrchestrator
from src.pipelines.data_integration import extract_modalities, load_records
from src.utils.explanations import compute_amount_stats


class ScoreRequest(BaseModel):
    records: List[Dict[str, Any]]
    train_on_payload: bool = False


class ScoreResponse(BaseModel):
    scores: List[float]
    investigation: Dict[str, Any]
    evaluation: Dict[str, Any]


app = FastAPI(title="AI Claims Fraud Detection API", version="1.0")

_config = load_config(os.getenv("FRAUD_CONFIG"))
_orchestrator = FraudOrchestrator(_config, prefer_spark=False)
_model_ready = False


def _bootstrap_model() -> None:
    global _model_ready
    if _model_ready:
        return

    sample_path = os.getenv("FRAUD_BOOTSTRAP_DATA", "data/sample/claims.jsonl")
    if not os.path.exists(sample_path):
        return

    records = load_records(sample_path, prefer_spark=False)
    modalities = extract_modalities(records, _config)
    labels = modalities.get("labels")
    if labels and all(label is not None for label in labels):
        _orchestrator.model.train(
            texts=modalities["texts"],
            image_paths=modalities["images"],
            structured_records=modalities["structured_records"],
            labels=labels,
        )
        _model_ready = True


@app.on_event("startup")
def startup() -> None:
    _bootstrap_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": _model_ready,
        "genai_enabled": _config.agents.genai_enabled,
        "genai_provider": _config.agents.genai_provider,
    }


@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest) -> Dict[str, Any]:
    if not payload.records:
        raise HTTPException(status_code=400, detail="No records provided")

    modalities = extract_modalities(payload.records, _config)
    labels = modalities.get("labels")

    if payload.train_on_payload:
        if not labels or any(label is None for label in labels):
            raise HTTPException(status_code=400, detail="Training requires labels in payload")
        _orchestrator.model.train(
            texts=modalities["texts"],
            image_paths=modalities["images"],
            structured_records=modalities["structured_records"],
            labels=labels,
        )
        global _model_ready
        _model_ready = True

    if not _model_ready:
        _bootstrap_model()

    scores = _orchestrator.model.predict_proba(
        texts=modalities["texts"],
        image_paths=modalities["images"],
        structured_records=modalities["structured_records"],
    )

    amount_stats = compute_amount_stats(modalities["records"])

    evaluation = _orchestrator.eval_agent.run({
        "labels": labels,
        "scores": scores.tolist(),
        "ids": modalities["ids"],
        "records": modalities["records"],
        "amount_stats": amount_stats,
    }).outputs

    investigation = _orchestrator.investigation_agent.run({
        "ids": modalities["ids"],
        "records": modalities["records"],
        "scores": scores.tolist(),
        "amount_stats": amount_stats,
    }).outputs

    return {
        "scores": scores.tolist(),
        "investigation": investigation,
        "evaluation": evaluation,
    }
