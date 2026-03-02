#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
ENDPOINT_ID="${VERTEX_ENDPOINT_ID:-}"
ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-fed-rate-endpoint}"

usage() {
  cat <<'EOF'
Usage:
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... scripts/vertex_endpoint_control.sh status
  PROJECT_ID=... REGION=... ENDPOINT_DISPLAY_NAME=... scripts/vertex_endpoint_control.sh create-endpoint
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... scripts/vertex_endpoint_control.sh undeploy-all
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... scripts/vertex_endpoint_control.sh undeploy
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... scripts/vertex_endpoint_control.sh off
  PROJECT_ID=... REGION=... VERTEX_ENDPOINT_ID=... MODEL_ID=... scripts/vertex_endpoint_control.sh deploy
  PROJECT_ID=... REGION=... MODEL_ID=... scripts/vertex_endpoint_control.sh deploy-auto

Optional deploy env vars:
  ENDPOINT_DISPLAY_NAME=fed-rate-endpoint
  DISPLAY_NAME=fed-rate-model
  MACHINE_TYPE=n1-standard-2
  TRAFFIC_SPLIT=0=100
  MIN_REPLICA_COUNT=1
  MAX_REPLICA_COUNT=1
EOF
}

require_base_env() {
  if [[ -z "$PROJECT_ID" || -z "$REGION" || -z "$ENDPOINT_ID" ]]; then
    echo "PROJECT_ID, REGION, and VERTEX_ENDPOINT_ID must be set." >&2
    exit 1
  fi
}

require_project_region() {
  if [[ -z "$PROJECT_ID" || -z "$REGION" ]]; then
    echo "PROJECT_ID and REGION must be set." >&2
    exit 1
  fi
}

endpoint_name() {
  echo "projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}"
}

create_endpoint() {
  require_project_region
  local endpoint_id
  endpoint_id="$(gcloud ai endpoints create \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --display-name="$ENDPOINT_DISPLAY_NAME" \
    --format="value(name.basename())")"

  if [[ -z "${endpoint_id// }" ]]; then
    echo "Failed to create endpoint." >&2
    exit 1
  fi

  ENDPOINT_ID="$endpoint_id"
  export VERTEX_ENDPOINT_ID="$ENDPOINT_ID"
  echo "Created endpoint: projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}"
  echo "Set VERTEX_ENDPOINT_ID=$ENDPOINT_ID for this shell session."
}

ensure_endpoint() {
  require_project_region
  if [[ -n "${ENDPOINT_ID// }" ]]; then
    return 0
  fi
  echo "VERTEX_ENDPOINT_ID is not set. Creating endpoint '${ENDPOINT_DISPLAY_NAME}'..."
  create_endpoint
}

status() {
  require_base_env
  echo "Endpoint: $(endpoint_name)"
  gcloud ai endpoints describe "$ENDPOINT_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --format="yaml(name,displayName,deployedModels)"
}

undeploy_all() {
  require_base_env
  local deployed_ids
  deployed_ids="$(gcloud ai endpoints describe "$ENDPOINT_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --format="value(deployedModels.id)")"

  if [[ -z "${deployed_ids// }" ]]; then
    echo "No deployed models found on endpoint $(endpoint_name)."
    return 0
  fi

  echo "Undeploying models from $(endpoint_name): $deployed_ids"
  for deployed_id in $deployed_ids; do
    gcloud ai endpoints undeploy-model "$ENDPOINT_ID" \
      --project="$PROJECT_ID" \
      --region="$REGION" \
      --deployed-model-id="$deployed_id" \
      --quiet
  done
  echo "Undeploy complete."
}

deploy() {
  ensure_endpoint
  local model_id="${MODEL_ID:-}"
  local display_name="${DISPLAY_NAME:-fed-rate-model}"
  local machine_type="${MACHINE_TYPE:-n1-standard-2}"
  local traffic_split="${TRAFFIC_SPLIT:-0=100}"
  local min_replicas="${MIN_REPLICA_COUNT:-1}"
  local max_replicas="${MAX_REPLICA_COUNT:-1}"

  if [[ -z "$model_id" ]]; then
    echo "MODEL_ID must be set for deploy." >&2
    exit 1
  fi

  echo "Deploying model ${model_id} to $(endpoint_name)"
  gcloud ai endpoints deploy-model "$ENDPOINT_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --model="$model_id" \
    --display-name="$display_name" \
    --machine-type="$machine_type" \
    --traffic-split="$traffic_split" \
    --min-replica-count="$min_replicas" \
    --max-replica-count="$max_replicas"
}

main() {
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud CLI is required." >&2
    exit 1
  fi

  local cmd="${1:-}"
  case "$cmd" in
    status) status ;;
    create-endpoint) create_endpoint ;;
    undeploy-all|undeploy|off) undeploy_all ;;
    deploy|deploy-auto) deploy ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "${1:-}"
