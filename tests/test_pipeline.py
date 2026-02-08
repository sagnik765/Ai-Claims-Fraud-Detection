from __future__ import annotations

from src.config import load_config
from src.orchestrator import FraudOrchestrator


def test_run_score():
    config = load_config(None)
    orchestrator = FraudOrchestrator(config)
    results = orchestrator.run("score", "data/sample/claims.jsonl")

    assert "scores" in results
    assert "evaluation" in results
    assert "investigation" in results
    assert "subrogation" in results
    assert len(results["scores"]) > 0
