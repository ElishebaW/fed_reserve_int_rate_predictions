#!/usr/bin/env python3
"""Minimal post-deploy smoke test for Vertex endpoint prediction spread."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any


SCENARIOS: list[dict[str, float]] = [
    {"Year": 2024, "Month": 1, "Day": 15, "Inflation Rate": 2.2, "Unemployment Rate": 4.7},
    {"Year": 2024, "Month": 4, "Day": 15, "Inflation Rate": 2.8, "Unemployment Rate": 4.2},
    {"Year": 2024, "Month": 7, "Day": 15, "Inflation Rate": 3.4, "Unemployment Rate": 3.9},
    {"Year": 2024, "Month": 10, "Day": 15, "Inflation Rate": 4.0, "Unemployment Rate": 3.5},
    {"Year": 2025, "Month": 1, "Day": 15, "Inflation Rate": 1.9, "Unemployment Rate": 5.1},
]


def to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unsupported prediction type: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Vertex endpoint prediction spread.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--std-threshold", type=float, default=0.05)
    parser.add_argument("--spread-threshold", type=float, default=0.10)
    parser.add_argument("--max-latency-ms", type=float, default=4000.0)
    args = parser.parse_args()

    try:
        from google.cloud import aiplatform
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "google-cloud-aiplatform is required. Install with: pip install -r requirements.txt"
        ) from exc

    aiplatform.init(project=args.project_id, location=args.region)
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{args.project_id}/locations/{args.region}/endpoints/{args.endpoint_id}"
    )

    predictions: list[float] = []
    latencies_ms: list[float] = []

    for index, scenario in enumerate(SCENARIOS, start=1):
        start = time.perf_counter()
        result = endpoint.predict(instances=[scenario])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        if not result.predictions:
            raise RuntimeError(f"Scenario {index} returned no predictions.")

        pred = to_float(result.predictions[0])
        predictions.append(pred)
        print(f"scenario_{index}: prediction={pred:.6f} latency_ms={elapsed_ms:.2f}")

    spread = max(predictions) - min(predictions)
    std_val = statistics.pstdev(predictions)
    p95_latency = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]

    report = {
        "scenario_count": len(SCENARIOS),
        "success_count": len(predictions),
        "predictions": predictions,
        "spread": spread,
        "std": std_val,
        "p95_latency_ms": p95_latency,
        "thresholds": {
            "std": args.std_threshold,
            "spread": args.spread_threshold,
            "max_latency_ms": args.max_latency_ms,
        },
    }
    print(json.dumps(report, indent=2))

    failures: list[str] = []
    if len(predictions) != len(SCENARIOS):
        failures.append("not all scenarios returned predictions")
    if std_val < args.std_threshold:
        failures.append(f"std too low: {std_val:.6f} < {args.std_threshold:.6f}")
    if spread < args.spread_threshold:
        failures.append(f"spread too low: {spread:.6f} < {args.spread_threshold:.6f}")
    if p95_latency > args.max_latency_ms:
        failures.append(f"p95 latency too high: {p95_latency:.2f} > {args.max_latency_ms:.2f}")

    if failures:
        print("SMOKE TEST FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
