# Agent Responsibilities

## Data Integration Agent
- Loads claims data from JSONL/CSV/Parquet
- Normalizes text fields and resolves image paths
- Produces clean multimodal payloads for modeling

## Investigation Agent
- Combines model score with fraud heuristics
- Emits flags and recommended investigation actions
- Generates GenAI-style rationales for review, including decline risk factors and claim amount context
- Optionally uses OpenAI to refine rationales (enabled via config + `OPENAI_API_KEY`)

## Evaluation Agent
- Computes metrics when labels are available
- Tracks confusion matrix and AUC
- Produces per-claim GenAI-style evaluation rationales when records are provided
- Optionally uses OpenAI to refine rationales (enabled via config + `OPENAI_API_KEY`)

## Subrogation/Salvage Agent
- Detects third-party liability and salvage signals
- Produces recovery and salvage indications
