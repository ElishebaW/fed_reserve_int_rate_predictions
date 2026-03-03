#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model}"
STD_THRESHOLD="${STD_THRESHOLD:-0.05}"
SPREAD_THRESHOLD="${SPREAD_THRESHOLD:-0.10}"
MAX_LATENCY_MS="${MAX_LATENCY_MS:-4000}"
SCHEMA_PATH="${SCHEMA_PATH:-$MODEL_DIR/feature_columns.json}"
MANIFEST_PATH="${MANIFEST_PATH:-$MODEL_DIR/model_manifest.json}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/model.joblib}"

CLI_PROJECT_ID="${PROJECT_ID:-}"
CLI_REGION="${REGION:-}"
CLI_ENDPOINT_ID="${VERTEX_ENDPOINT_ID:-}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PROJECT_ID="${CLI_PROJECT_ID:-${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}}"
REGION="${CLI_REGION:-${REGION:-us-central1}}"
ENDPOINT_ID="${CLI_ENDPOINT_ID:-${VERTEX_ENDPOINT_ID:-}}"

usage() {
  cat <<EOF
Usage:
  scripts/release_gate.sh precheck
  PROJECT_ID=... REGION=... MODEL_ID=... scripts/release_gate.sh deploy
  PROJECT_ID=... REGION=... MODEL_DISPLAY_NAME=... scripts/release_gate.sh deploy
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... scripts/release_gate.sh smoke [ENDPOINT_ID]
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... scripts/release_gate.sh run [ENDPOINT_ID]
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... scripts/release_gate.sh undeploy [ENDPOINT_ID]

Commands:
  precheck   Run artifact + schema gates locally.
  deploy     Auto-create endpoint if missing, then deploy model.
  smoke      Run endpoint smoke test thresholds.
  run        Run precheck, then smoke (full release gate).
  undeploy   One-command endpoint undeploy (cost control).

Threshold env vars:
  STD_THRESHOLD (default: ${STD_THRESHOLD})
  SPREAD_THRESHOLD (default: ${SPREAD_THRESHOLD})
  MAX_LATENCY_MS (default: ${MAX_LATENCY_MS})

Deploy env vars:
  MODEL_ID=...
  MODEL_DISPLAY_NAME=...   # if MODEL_ID is unset, latest model for this display name is used
  ENDPOINT_DISPLAY_NAME=...  # used when creating a new endpoint
  DISPLAY_NAME=...           # deployed model display name (default: fed-rate-model)
  MACHINE_TYPE=...           # default: n1-standard-2
  MIN_REPLICA_COUNT=...      # default: 1
  MAX_REPLICA_COUNT=...      # default: 1
  TRAFFIC_SPLIT=...          # default: 0=100
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

sha256_file() {
  local path="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
    return 0
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
    return 0
  fi
  echo "Neither shasum nor sha256sum is available." >&2
  exit 1
}

artifact_gate() {
  echo "[gate] Artifact integrity"
  [[ -f "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
  [[ -f "$SCHEMA_PATH" ]] || { echo "Missing schema: $SCHEMA_PATH" >&2; exit 1; }
  [[ -f "$MANIFEST_PATH" ]] || { echo "Missing manifest: $MANIFEST_PATH" >&2; exit 1; }

  local expected actual
  expected="$(python3 - <<'PY' "$MANIFEST_PATH"
import json
import sys
manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
sha = payload.get("artifacts", {}).get("model_joblib_sha256")
if not isinstance(sha, str) or not sha:
    raise SystemExit(1)
print(sha)
PY
)" || { echo "Manifest missing artifacts.model_joblib_sha256" >&2; exit 1; }

  actual="$(sha256_file "$MODEL_PATH")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for model.joblib" >&2
    echo "Expected: $expected" >&2
    echo "Actual:   $actual" >&2
    exit 1
  fi
  echo "[ok] Artifact checksum matches manifest"
}

schema_gate() {
  echo "[gate] Schema contract"
  python3 - <<'PY' "$SCHEMA_PATH"
import json
import sys

schema_path = sys.argv[1]
required = {"Year", "Month", "Day", "Inflation Rate", "Unemployment Rate"}

with open(schema_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

cols = payload.get("feature_columns")
if not isinstance(cols, list) or not cols:
    raise SystemExit("feature_columns must be a non-empty list")
if any(not isinstance(c, str) or not c.strip() for c in cols):
    raise SystemExit("feature_columns must contain only non-empty strings")
if len(cols) != len(set(cols)):
    raise SystemExit("feature_columns contains duplicates")

missing = sorted(required - set(cols))
if missing:
    raise SystemExit(f"missing required features: {missing}")

print("[ok] Schema gate passed")
PY
}

require_endpoint_env() {
  local missing=()
  [[ -z "$PROJECT_ID" ]] && missing+=("PROJECT_ID")
  [[ -z "$REGION" ]] && missing+=("REGION")
  [[ -z "$ENDPOINT_ID" ]] && missing+=("VERTEX_ENDPOINT_ID")
  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "Missing required env vars: ${missing[*]}" >&2
    echo "Tip: export vars or pass inline with the command." >&2
    echo "Example: PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... scripts/release_gate.sh undeploy" >&2
    echo "Or: scripts/release_gate.sh undeploy <ENDPOINT_ID>" >&2
    exit 1
  fi
}

smoke_gate() {
  require_endpoint_env
  echo "[gate] Smoke test (endpoint)"
  python3 "$ROOT_DIR/scripts/smoke_test_vertex_endpoint.py" \
    --project-id "$PROJECT_ID" \
    --region "$REGION" \
    --endpoint-id "$ENDPOINT_ID" \
    --std-threshold "$STD_THRESHOLD" \
    --spread-threshold "$SPREAD_THRESHOLD" \
    --max-latency-ms "$MAX_LATENCY_MS"
}

undeploy_now() {
  require_endpoint_env
  echo "[action] Undeploy endpoint models"
  PROJECT_ID="$PROJECT_ID" REGION="$REGION" VERTEX_ENDPOINT_ID="$ENDPOINT_ID" \
    "$ROOT_DIR/scripts/vertex_endpoint_control.sh" undeploy
}

resolve_model_id() {
  local display_name="${1:-}"
  local filter=""
  if [[ -n "$display_name" ]]; then
    filter="displayName=\"${display_name}\""
  fi

  gcloud ai models list \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    ${filter:+--filter="$filter"} \
    --sort-by="~createTime" \
    --limit=1 \
    --format="value(name.basename())"
}

deploy_now() {
  require_cmd gcloud
  if [[ -z "$PROJECT_ID" || -z "$REGION" ]]; then
    echo "PROJECT_ID and REGION are required." >&2
    exit 1
  fi

  local model_id="${MODEL_ID:-}"
  if [[ -z "${model_id// }" ]]; then
    model_id="$(resolve_model_id "${MODEL_DISPLAY_NAME:-}" || true)"
  fi
  if [[ -z "${model_id// }" ]]; then
    echo "Unable to resolve MODEL_ID. Set MODEL_ID or MODEL_DISPLAY_NAME and rerun." >&2
    exit 1
  fi

  local endpoint_id="${ENDPOINT_ID:-}"
  if [[ -z "${endpoint_id// }" ]]; then
    endpoint_id="$(gcloud ai endpoints create \
      --project="$PROJECT_ID" \
      --region="$REGION" \
      --display-name="${ENDPOINT_DISPLAY_NAME:-fed-rate-endpoint}" \
      --format="value(name.basename())")"
    if [[ -z "${endpoint_id// }" ]]; then
      echo "Failed to create endpoint." >&2
      exit 1
    fi
  fi

  echo "[info] MODEL_ID=${model_id}"
  echo "[info] VERTEX_ENDPOINT_ID=${endpoint_id}"
  echo "[action] Deploy model to endpoint"
  gcloud ai endpoints deploy-model "$endpoint_id" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --model="$model_id" \
    --display-name="${DISPLAY_NAME:-fed-rate-model}" \
    --machine-type="${MACHINE_TYPE:-n1-standard-2}" \
    --min-replica-count="${MIN_REPLICA_COUNT:-1}" \
    --max-replica-count="${MAX_REPLICA_COUNT:-1}" \
    --traffic-split="${TRAFFIC_SPLIT:-0=100}"

  ENDPOINT_ID="$endpoint_id"
  export VERTEX_ENDPOINT_ID="$ENDPOINT_ID"
  echo "[done] Exported VERTEX_ENDPOINT_ID=${VERTEX_ENDPOINT_ID}"
}

main() {
  local cmd="${1:-}"
  local endpoint_arg="${2:-}"
  if [[ -n "$endpoint_arg" && -z "$ENDPOINT_ID" ]]; then
    ENDPOINT_ID="$endpoint_arg"
    export VERTEX_ENDPOINT_ID="$ENDPOINT_ID"
  fi

  case "$cmd" in
    precheck)
      require_cmd python3
      artifact_gate
      schema_gate
      echo "RELEASE GATE PRECHECK PASSED"
      ;;
    deploy)
      deploy_now
      ;;
    smoke)
      require_cmd python3
      smoke_gate
      ;;
    run)
      require_cmd python3
      artifact_gate
      schema_gate
      smoke_gate
      echo "RELEASE GATE PASSED"
      ;;
    undeploy)
      undeploy_now
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "${1:-}"
