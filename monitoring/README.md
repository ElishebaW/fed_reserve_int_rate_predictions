# Vertex Monitoring Setup

## 1) Create model monitoring job
```bash
gcloud ai model-monitoring-jobs create \
  --location=us-central1 \
  --display-name=fed-rate-endpoint-monitor \
  --config-from-file=monitoring/model_monitoring_config.yaml
```

## 2) Create latency alert policy
```bash
gcloud alpha monitoring policies create \
  --policy-from-file=monitoring/latency_alert_policy.json
```

## 3) Verify telemetry
- Confirm custom app metrics are flowing:
  - `custom.googleapis.com/genai/fed_rate_copilot/prediction_success`
  - `custom.googleapis.com/genai/fed_rate_copilot/missing_pct`
  - `custom.googleapis.com/genai/fed_rate_copilot/total_latency_ms`
- Confirm endpoint latency metric is populated:
  - `aiplatform.googleapis.com/endpoint/prediction_latencies`
