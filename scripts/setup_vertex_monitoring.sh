#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITOR_DIR="$ROOT_DIR/monitoring"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-us-central1}"
ENDPOINT_ID="${VERTEX_ENDPOINT_ID:-${ENDPOINT_ID:-}}"
TRAINING_BASELINE_URI="${TRAINING_BASELINE_URI:-}"
BASELINE_FORMAT="${BASELINE_FORMAT:-jsonl}"
ALERT_EMAILS="${ALERT_EMAILS:-}"
MONITOR_JOB_DISPLAY_NAME="${MONITOR_JOB_DISPLAY_NAME:-fed-rate-endpoint-monitor}"
PREDICTION_SAMPLING_RATE="${PREDICTION_SAMPLING_RATE:-0.30}"
MONITORING_FREQUENCY_HOURS="${MONITORING_FREQUENCY_HOURS:-24}"
NOTIFICATION_CHANNEL="${NOTIFICATION_CHANNEL:-}"

usage() {
  cat <<EOF
Usage:
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... TRAINING_BASELINE_URI=gs://... ALERT_EMAILS=you@company.com \\
  scripts/setup_vertex_monitoring.sh

Optional env vars:
  BASELINE_FORMAT=jsonl|csv|tf-record|tf-record-gzip|bigquery
  MONITOR_JOB_DISPLAY_NAME=fed-rate-endpoint-monitor
  PREDICTION_SAMPLING_RATE=0.30
  MONITORING_FREQUENCY_HOURS=24
  NOTIFICATION_CHANNEL=projects/PROJECT_ID/notificationChannels/CHANNEL_ID
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

require_env() {
  local missing=()
  [[ -z "$PROJECT_ID" ]] && missing+=("PROJECT_ID")
  [[ -z "$REGION" ]] && missing+=("REGION")
  [[ -z "$ENDPOINT_ID" ]] && missing+=("VERTEX_ENDPOINT_ID")
  [[ -z "$TRAINING_BASELINE_URI" ]] && missing+=("TRAINING_BASELINE_URI")
  [[ -z "$ALERT_EMAILS" ]] && missing+=("ALERT_EMAILS")
  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "Missing required env vars: ${missing[*]}" >&2
    usage
    exit 1
  fi
}

render_template() {
  local src="$1"
  local dst="$2"
  sed \
    -e "s|__DEPLOYED_MODEL_ID__|$DEPLOYED_MODEL_ID|g" \
    -e "s|__TRAINING_BASELINE_URI__|$TRAINING_BASELINE_URI|g" \
    -e "s|__BASELINE_FORMAT__|$BASELINE_FORMAT|g" \
    -e "s|__ENDPOINT_ID__|$ENDPOINT_ID|g" \
    "$src" > "$dst"
}

policy_exists() {
  local display_name="$1"
  local existing
  existing="$(gcloud monitoring policies list \
    --project="$PROJECT_ID" \
    --filter="displayName=\"${display_name}\"" \
    --limit=1 \
    --format="value(name)")"
  [[ -n "${existing// }" ]]
}

create_policy_if_missing() {
  local policy_file="$1"
  local display_name
  display_name="$(python3 - <<'PY' "$policy_file"
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
print(payload["displayName"])
PY
)"

  if policy_exists "$display_name"; then
    echo "[skip] Alert policy already exists: $display_name"
    return 0
  fi

  local cmd=(
    gcloud monitoring policies create
    --project="$PROJECT_ID"
    --policy-from-file="$policy_file"
  )
  if [[ -n "${NOTIFICATION_CHANNEL// }" ]]; then
    cmd+=(--notification-channels="$NOTIFICATION_CHANNEL")
  fi

  echo "[create] Alert policy: $display_name"
  "${cmd[@]}"
}

main() {
  require_cmd gcloud
  require_cmd python3
  require_env

  DEPLOYED_MODEL_ID="$(gcloud ai endpoints describe "$ENDPOINT_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --format="value(deployedModels[0].id)")"
  if [[ -z "${DEPLOYED_MODEL_ID// }" ]]; then
    echo "No deployed model found on endpoint '$ENDPOINT_ID'. Deploy a model first." >&2
    exit 1
  fi

  tmp_objective="$(mktemp)"
  tmp_latency="$(mktemp)"
  tmp_error="$(mktemp)"
  trap 'rm -f "$tmp_objective" "$tmp_latency" "$tmp_error"' EXIT

  render_template "$MONITOR_DIR/model_monitoring_objective_config.yaml.template" "$tmp_objective"
  render_template "$MONITOR_DIR/latency_alert_policy.json.template" "$tmp_latency"
  render_template "$MONITOR_DIR/error_spike_alert_policy.json.template" "$tmp_error"

  echo "[info] PROJECT_ID=$PROJECT_ID"
  echo "[info] REGION=$REGION"
  echo "[info] VERTEX_ENDPOINT_ID=$ENDPOINT_ID"
  echo "[info] DEPLOYED_MODEL_ID=$DEPLOYED_MODEL_ID"

  echo "[create] Model monitoring job: $MONITOR_JOB_DISPLAY_NAME"
  gcloud ai model-monitoring-jobs create \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --display-name="$MONITOR_JOB_DISPLAY_NAME" \
    --endpoint="$ENDPOINT_ID" \
    --prediction-sampling-rate="$PREDICTION_SAMPLING_RATE" \
    --monitoring-frequency="$MONITORING_FREQUENCY_HOURS" \
    --emails="$ALERT_EMAILS" \
    --monitoring-config-from-file="$tmp_objective"

  create_policy_if_missing "$tmp_latency"
  create_policy_if_missing "$tmp_error"

  echo "[done] Vertex monitoring + alert policies configured."
}

main "$@"
