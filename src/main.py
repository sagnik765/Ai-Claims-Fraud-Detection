from __future__ import annotations

import argparse
import json

from src.config import load_config
from src.orchestrator import FraudOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P&C Fraud Detection Agentic Pipeline")
    parser.add_argument("--mode", choices=["train", "score"], default="score")
    parser.add_argument("--data", required=True, help="Path to claims data (jsonl/csv/parquet)")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--prefer-spark", action="store_true", help="Use Spark if available")
    parser.add_argument("--output", default=None, help="Write results to JSON file")
    parser.add_argument("--model-in", default=None, help="Load a saved model artifact")
    parser.add_argument("--model-out", default=None, help="Save model artifact after training")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    orchestrator = FraudOrchestrator(config, prefer_spark=args.prefer_spark)
    if args.model_in:
        orchestrator.model = orchestrator.model.load(args.model_in)
    results = orchestrator.run(args.mode, args.data)

    if args.mode == "train" and args.model_out:
        orchestrator.model.save(args.model_out)

    payload = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
