# Production Execution Plan (No Silent Failures)

## Phase 1: Contract lock
1. Freeze `feature_columns.json` and publish version tag (for both extractor and predictor).
2. Enforce strict extraction schema (`features`, `missing_features`, `assumptions`, `clarifying_question`).
3. Reject inference requests with missing or unexpected keys.
4. Remove all automatic imputation and heuristic fallback paths.

Exit criteria:
- Any missing feature produces explicit blocking error.
- Predictor returns deterministic error codes (`INVALID_INPUT`, `MODEL_NOT_READY`, `PREDICT_FAILED`).

## Phase 2: Service hardening
1. Route all Vertex calls through `inference_service.py`.
2. Set fixed request timeout (20s) and explicit timeout error handling.
3. Parse Vertex payload strictly and reject unknown prediction shapes.
4. Add structured logs with interaction id and failure class.

Exit criteria:
- Timeout, schema mismatch, and payload errors are separately observable.

## Phase 3: Test + eval gate
1. Run `pytest` on every PR and block merge on failure.
2. Run `evals/run_llm_judge_eval.py` on extraction cases before deploy.
3. Set release gate thresholds:
   - Unit tests: 100% pass
   - LLM judge extraction pass rate: >= 95%
   - No critical schema violations

Exit criteria:
- Build fails if eval gate thresholds are not met.

## Phase 4: Monitoring + alerting
1. Enable Vertex Model Monitoring job (`monitoring/model_monitoring_config.yaml`).
2. Create latency alert policy (`monitoring/latency_alert_policy.json`).
3. Publish app-level metrics (`prediction_success`, `missing_pct`, `total_latency_ms`).
4. Wire alerts to on-call channel.

Exit criteria:
- Drift/skew and P95 latency alerts are active in production.

## Phase 5: Deploy and rollout
1. Deploy predictor and Streamlit revisions to staging.
2. Run smoke tests with strict contract payloads only.
3. Promote to production with canary traffic split (10% -> 50% -> 100%).
4. Monitor for 24h post-rollout and auto-rollback on alert thresholds.

Exit criteria:
- Stable error rate and latency under SLO for 24h.

## SLO targets
- Availability: 99.9%
- P95 inference latency: < 2.0s
- Extraction schema compliance: 100%
- Silent failure rate: 0%
