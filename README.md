# P&C Fraud Detection - Agentic Multimodal System

This project is a complete, runnable scaffold for detecting fraudulent claims in Property & Casualty insurance using multimodal data (text + images), big-data integration patterns, and an agentic AI workflow.

## What is included
- Multimodal model pipeline (text + image features)
- Agentic workflow with investigation, evaluation, subrogation/salvage indication, and data integration agents
- Big-data integration hooks (Spark optional) with local fallback
- Sample dataset and schema
- CLI and scripts to train and score

## Quick start
1. Create a virtual environment
2. Install dependencies
3. Run the pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --mode train --data data/sample/claims.jsonl --model-out artifacts/model.pkl
python -m src.main --mode score --data data/sample/claims.jsonl --model-in artifacts/model.pkl
```

## Using your dataset
This repo supports arbitrary tabular claim datasets. The structured featurizer automatically encodes numeric and categorical columns, while text/image are optional.

For the dataset located at `/Users/sagnikroy/Downloads/Insurance claims data.csv`, use the provided config:

```bash
python -m src.main --mode train --data \"/Users/sagnikroy/Downloads/Insurance claims data.csv\" --config config_claims_data.yaml --model-out artifacts/claims_model.pkl --prefer-spark
python -m src.main --mode score --data \"/Users/sagnikroy/Downloads/Insurance claims data.csv\" --config config_claims_data.yaml --model-in artifacts/claims_model.pkl --prefer-spark
```

## Project structure
- `src/agents`: agent implementations
- `src/models`: multimodal model
- `src/pipelines`: data integration
- `src/utils`: utilities and helpers
- `data/sample`: sample claims data
- `docs`: architecture and schema
- `scripts`: convenience scripts

## Notes
- This is a baseline scaffold built to be extended with your data lake, image stores, and model infrastructure.
- The code runs without Spark or deep learning libraries, but will use them if installed.
