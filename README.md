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

## GenAI rationales (OpenAI)
To generate OpenAI-backed rationales in the investigation and evaluation agents:

1. Install the OpenAI client (already in `requirements.txt`)
2. Set your API key:
```bash
export OPENAI_API_KEY="YOUR_KEY"
```
3. Enable GenAI in config:
```yaml
agents:
  genai_enabled: true
  genai_provider: openai
  genai_model: gpt-5.2
  genai_max_claims: 50
  genai_min_score: 0.6
  genai_scope: high_risk
```

If disabled or no key is set, the system falls back to template-based rationales.

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
