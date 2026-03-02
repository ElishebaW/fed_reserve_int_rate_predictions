# Deploy Checklist: Gate and Smoke Acceptance Criteria

Use this checklist before promoting any new model artifact bundle to production.

## 1) Artifact Integrity Gate (must pass all)
- `model/model.joblib` exists.
- `model/feature_columns.json` exists.
- `model/model_manifest.json` exists.
- `model_manifest.json` includes `artifacts.model_joblib_sha256`.
- Local `sha256(model/model.joblib)` exactly matches `artifacts.model_joblib_sha256`.

Fail condition:
- Any file missing, manifest key missing, or checksum mismatch.

## 2) Schema Contract Gate (must pass all)
- `feature_columns.json` contains a non-empty `feature_columns` list.
- No duplicate feature names.
- Predictor accepts all listed features and rejects unknown features.
- Current mandatory features are present: `Year`, `Month`, `Day`, `Inflation Rate`, `Unemployment Rate`.

Fail condition:
- Missing mandatory fields, empty schema, or parsing contract violation.

## 3) Sensitivity Gate (must pass all)
Run fixed-scenario smoke suite against deployed endpoint.

Thresholds:
- `scenario_count >= 5`
- `success_count == scenario_count`
- prediction spread (`max - min`) `>= 0.10`
- prediction standard deviation (`std`) `>= 0.05`

Rationale:
- Prevents recurrence of collapsed/near-constant predictions.

Fail condition:
- Any threshold not met.

## 4) Runtime Reliability Gate (must pass all)
- Endpoint health check returns success.
- No 4xx/5xx prediction errors in smoke run.
- p95 prediction latency for smoke run `<= 4000 ms`.

Fail condition:
- Health failure, error responses, or p95 above threshold.

## 5) Promotion Decision
Promote only if all four gates pass.

If failed:
1. Do not promote.
2. Keep current deployed model.
3. Record failure reason and rerun after fix.
