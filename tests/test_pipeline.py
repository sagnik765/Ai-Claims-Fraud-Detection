from __future__ import annotations

import json

from src.config import load_config
from src.orchestrator import FraudOrchestrator


def _write_sample(path):
    samples = [
        {
            "claim_id": "C-1",
            "claim_description": "Rear-ended at a stop light",
            "loss_description": "Rear bumper damage",
            "adjuster_notes": "Police report obtained",
            "image_paths": [],
            "claim_amount": 4200,
            "policy_age_days": 180,
            "prior_claims_count": 0,
            "late_reported": 0,
            "multiple_parties": 1,
            "injury_reported": 0,
            "total_loss": 0,
            "is_fraud": 0,
        },
        {
            "claim_id": "C-2",
            "claim_description": "Hit and run overnight",
            "loss_description": "Front damage and possible frame issues",
            "adjuster_notes": "Reported 45 days after incident",
            "image_paths": [],
            "claim_amount": 11500,
            "policy_age_days": 12,
            "prior_claims_count": 2,
            "late_reported": 1,
            "multiple_parties": 0,
            "injury_reported": 0,
            "total_loss": 1,
            "is_fraud": 1,
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row) + "\n")


def test_run_score(tmp_path):
    config = load_config(None)
    orchestrator = FraudOrchestrator(config)
    data_path = tmp_path / "claims.jsonl"
    _write_sample(data_path)
    results = orchestrator.run("score", str(data_path))

    assert "scores" in results
    assert "evaluation" in results
    assert "investigation" in results
    assert "subrogation" in results
    assert len(results["scores"]) > 0
