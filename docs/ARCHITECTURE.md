# Architecture Overview

This project is structured as an agentic, multimodal fraud detection system for P&C claims. It supports text and image inputs, scalable data ingestion, and a multi-agent workflow to assist investigators.

## Layers
- Data integration layer
- Feature extraction and modeling layer
- Agentic workflow layer
- Delivery and monitoring layer

## Data integration layer
- Text and image signals are ingested from data lakes, claim systems, and media stores.
- Optional Spark integration enables batch and distributed data processing.
- Local fallback uses pandas or JSONL for rapid iteration.

## Feature extraction and modeling
- Text features use TF-IDF or hashing as a baseline.
- Image features use statistical descriptors or can be replaced by deep CNN embeddings.
- Structured claim attributes are combined with text and image features.

## Agentic workflow
- Data integration agent: assembles multimodal inputs for scoring.
- Investigation agent: produces fraud flags and recommended actions.
- Evaluation agent: tracks model performance and drift metrics.
- Subrogation/salvage agent: identifies recovery and salvage indications.

## Deployment options
- Batch scoring through scheduled jobs or Spark pipelines.
- Near-real-time scoring through APIs or event streaming.
- Experiment tracking via MLflow and model registry integration.
