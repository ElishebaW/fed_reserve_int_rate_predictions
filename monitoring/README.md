# Vertex Monitoring Setup (Model Monitoring v1)

Use CLI for repeatable setup. UI is still useful for creating notification channels if your
`gcloud` install does not include monitoring channel commands.

## What gets configured
- Drift + skew (distribution shift) monitoring job for deployed model inputs.
- Alert policy for endpoint latency spikes.
- Alert policy for prediction success drop (error spikes proxy from app telemetry).

## Required inputs
- `PROJECT_ID`
- `REGION` (default: `us-central1`)
- `VERTEX_ENDPOINT_ID`
- `TRAINING_BASELINE_URI` (for skew baseline, for example `gs://.../training_baseline.jsonl`)
- `ALERT_EMAILS` (comma-separated)

Optional:
- `NOTIFICATION_CHANNEL` in format:
  `projects/PROJECT_ID/notificationChannels/CHANNEL_ID`

## One-command setup
```bash
PROJECT_ID="fed-rate-prediction" \
REGION="us-central1" \
VERTEX_ENDPOINT_ID="<ENDPOINT_ID>" \
TRAINING_BASELINE_URI="gs://<BUCKET>/monitoring/training_baseline.jsonl" \
ALERT_EMAILS="you@company.com" \
bash scripts/setup_vertex_monitoring.sh
```

## Optional tuning knobs
```bash
MONITOR_JOB_DISPLAY_NAME="fed-rate-endpoint-monitor"
PREDICTION_SAMPLING_RATE="0.30"
MONITORING_FREQUENCY_HOURS="24"
BASELINE_FORMAT="jsonl"
DISPLAY_NAME="fed-rate-model"
```

## Files
- `monitoring/model_monitoring_objective_config.yaml.template`
- `monitoring/latency_alert_policy.json.template`
- `monitoring/error_spike_alert_policy.json.template`
- `scripts/setup_vertex_monitoring.sh`

## Verify telemetry
- Vertex latency metric:
  - `aiplatform.googleapis.com/endpoint/prediction_latencies`
- App custom metrics:
  - `custom.googleapis.com/genai/fed_rate_copilot/prediction_success`
  - `custom.googleapis.com/genai/fed_rate_copilot/missing_pct`
  - `custom.googleapis.com/genai/fed_rate_copilot/total_latency_ms`
