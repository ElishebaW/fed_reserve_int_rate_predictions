import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from google.cloud import storage

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
MAX_INSTANCES = int(os.environ.get("MAX_PREDICT_INSTANCES", "128"))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH_BYTES", str(1_000_000)))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
model = None
feature_columns: list[str] = []


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    no_scheme = uri.replace("gs://", "", 1)
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")


def _download_from_aip_storage_uri() -> None:
    aip_storage_uri = os.environ.get("AIP_STORAGE_URI", "")
    if not aip_storage_uri.startswith("gs://"):
        return

    bucket_name, prefix = _split_gcs_uri(aip_storage_uri)
    if prefix:
        model_blob_path = f"{prefix}/model.joblib"
        schema_blob_path = f"{prefix}/feature_columns.json"
    else:
        model_blob_path = "model.joblib"
        schema_blob_path = "feature_columns.json"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    bucket.blob(model_blob_path).download_to_filename(MODEL_PATH)
    bucket.blob(schema_blob_path).download_to_filename(SCHEMA_PATH)


def load_artifacts() -> None:
    global model
    global feature_columns

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCHEMA_PATH):
        _download_from_aip_storage_uri()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Feature schema not found: {SCHEMA_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    cols = schema.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError("Invalid feature_columns schema.")
    feature_columns = cols


def parse_instances(payload: dict[str, Any]) -> pd.DataFrame:
    if "instances" not in payload:
        raise ValueError("Request must include 'instances'.")

    instances = payload["instances"]
    if not isinstance(instances, list) or len(instances) == 0:
        raise ValueError("'instances' must be a non-empty list.")
    if len(instances) > MAX_INSTANCES:
        raise ValueError(f"Too many instances. Max allowed is {MAX_INSTANCES}.")

    frame = pd.DataFrame(instances)

    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    unexpected = [c for c in frame.columns if c not in feature_columns]
    if unexpected:
        raise ValueError(f"Unexpected feature columns: {unexpected}")

    ordered = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if ordered.isna().any().any():
        raise ValueError("All features must be numeric and non-null.")

    values = ordered.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All feature values must be finite numbers.")

    return ordered


@app.get("/health")
def health():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("Invalid JSON payload.")
        x = parse_instances(payload)
        preds = model.predict(x)
        return jsonify({"predictions": preds.tolist()})
    except Exception:
        return jsonify({"error": "Invalid prediction request."}), 400


if __name__ == "__main__":
    load_artifacts()
    port = int(os.environ.get("AIP_HTTP_PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
