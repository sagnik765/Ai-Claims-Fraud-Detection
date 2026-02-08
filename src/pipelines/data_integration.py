from __future__ import annotations

from typing import Any, Dict, List, Optional
import csv
import json
import os
import warnings

from src.config import AppConfig
from src.utils.optional import optional_import
from src.utils.text import join_fields


SupportedRecord = Dict[str, Any]


def _load_jsonl(path: str) -> List[SupportedRecord]:
    records: List[SupportedRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_csv(path: str) -> List[SupportedRecord]:
    records: List[SupportedRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def _load_parquet_pandas(path: str) -> List[SupportedRecord]:
    pandas, err = optional_import("pandas")
    if pandas is None:
        raise RuntimeError("pandas is required to read parquet locally") from err
    df = pandas.read_parquet(path)
    return df.to_dict(orient="records")


def _load_with_spark(path: str) -> Optional[List[SupportedRecord]]:
    pyspark, err = optional_import("pyspark")
    if pyspark is None:
        return None

    from pyspark.sql import SparkSession  # type: ignore

    spark = SparkSession.builder.appName("fraud-data-integration").getOrCreate()
    _, ext = os.path.splitext(path.lower())

    if ext == ".parquet":
        df = spark.read.parquet(path)
    elif ext == ".csv":
        df = spark.read.option("header", True).csv(path)
    elif ext in (".json", ".jsonl"):
        df = spark.read.json(path)
    else:
        raise ValueError(f"Unsupported file extension for spark: {ext}")

    # Convert to local records for this baseline scaffold
    return [json.loads(row) for row in df.toJSON().collect()]


def load_records(path: str, prefer_spark: bool = False) -> List[SupportedRecord]:
    if prefer_spark:
        try:
            records = _load_with_spark(path)
            if records is not None:
                return records
        except Exception as exc:  # pragma: no cover - spark is optional
            warnings.warn(f"Spark load failed, falling back to local reader: {exc}")

    _, ext = os.path.splitext(path.lower())
    if ext in (".jsonl", ".json"):
        return _load_jsonl(path)
    if ext == ".csv":
        return _load_csv(path)
    if ext == ".parquet":
        return _load_parquet_pandas(path)

    raise ValueError(f"Unsupported file extension: {ext}")


def extract_modalities(records: List[SupportedRecord], config: AppConfig) -> Dict[str, Any]:
    texts: List[str] = []
    images: List[List[str]] = []
    labels: List[Optional[int]] = []
    ids: List[str] = []
    structured_records: List[SupportedRecord] = []

    for idx, record in enumerate(records):
        claim_id = str(record.get("claim_id", idx))
        ids.append(claim_id)

        text_values = [str(record.get(f, "")) for f in config.data.text_fields]
        texts.append(join_fields(text_values))

        image_field = record.get(config.data.image_field, [])
        if isinstance(image_field, str):
            image_paths = [p.strip() for p in image_field.split(";") if p.strip()]
        else:
            image_paths = list(image_field) if image_field else []
        images.append(image_paths)

        label_value = record.get(config.data.label_field)
        if label_value is None or label_value == "":
            labels.append(None)
        else:
            labels.append(int(label_value))

        exclude_keys = set(config.data.text_fields)
        exclude_keys.add(config.data.image_field)
        exclude_keys.add(config.data.label_field)
        exclude_keys.update(["claim_id", "policy_id"])
        structured_records.append({k: v for k, v in record.items() if k not in exclude_keys})

    return {
        "ids": ids,
        "texts": texts,
        "images": images,
        "labels": labels,
        "records": records,
        "structured_records": structured_records,
    }
