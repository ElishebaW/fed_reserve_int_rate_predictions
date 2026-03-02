#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model}"
STD_THRESHOLD="${STD_THRESHOLD:-0.05}"
SPREAD_THRESHOLD="${SPREAD_THRESHOLD:-0.10}"
MAX_LATENCY_MS="${MAX_LATENCY_MS:-4000}"
MODEL_IMAGE_HINT="${MODEL_IMAGE_HINT:-fed-rate}"
SCHEMA_PATH="${SCHEMA_PATH:-$MODEL_DIR/feature_columns.json}"
MANIFEST_PATH="${MANIFEST_PATH:-$MODEL_DIR/model_manifest.json}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/model.joblib}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-us-central1}"
ENDPOINT_ID="${VERTEX_ENDPOINT_ID:-}"

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
  MODEL_DISPLAY_NAME=...   # if MODEL_ID is unset, latest match is used
  MODEL_FILTER=...         # optional extra gcloud filter when auto-selecting latest
  MODEL_IMAGE_HINT=...     # default: fed-rate, matched against container image URI
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
  if [[ -z "$PROJECT_ID" || -z "$REGION" || -z "$ENDPOINT_ID" ]]; then
    echo "PROJECT_ID, REGION, and VERTEX_ENDPOINT_ID are required." >&2
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
  "$ROOT_DIR/scripts/vertex_endpoint_control.sh" undeploy
}

resolve_model_id() {
  local list_filter="${1:-}"
  local image_hint="${2:-}"
  local candidate_ids image_uri

  if [[ -n "$list_filter" ]]; then
    candidate_ids="$(gcloud ai models list \
      --project="$PROJECT_ID" \
      --region="$REGION" \
      --filter="$list_filter" \
      --sort-by="~createTime" \
      --limit=50 \
      --format="value(name.basename())")"
  else
    candidate_ids="$(gcloud ai models list \
      --project="$PROJECT_ID" \
      --region="$REGION" \
      --sort-by="~createTime" \
      --limit=50 \
      --format="value(name.basename())")"
  fi

  for candidate in $candidate_ids; do
    image_uri="$(gcloud ai models describe "$candidate" \
      --project="$PROJECT_ID" \
      --region="$REGION" \
      --format="value(containerSpec.imageUri)" 2>/dev/null || true)"
    if [[ -z "$image_hint" ]]; then
      echo "$candidate"
      return 0
    fi
    if [[ "$image_uri" == *"$image_hint"* ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

deploy_now() {
  require_cmd gcloud
  if [[ -z "$PROJECT_ID" || -z "$REGION" ]]; then
    echo "PROJECT_ID and REGION are required." >&2
    exit 1
  fi

  local model_id="${MODEL_ID:-}"
  if [[ -z "$model_id" ]]; then
    local display_name="${MODEL_DISPLAY_NAME:-}"
    local extra_filter="${MODEL_FILTER:-}"
    local filter=""
    if [[ -n "$display_name" ]]; then
      filter="displayName=\"${display_name}\""
      if [[ -n "$extra_filter" ]]; then
        filter="${filter} AND (${extra_filter})"
      fi
      model_id="$(resolve_model_id "$filter" "$MODEL_IMAGE_HINT" || true)"
      if [[ -z "${model_id// }" ]]; then
        echo "No matching custom model found for MODEL_DISPLAY_NAME='${display_name}' with MODEL_IMAGE_HINT='${MODEL_IMAGE_HINT}' in ${PROJECT_ID}/${REGION}." >&2
        echo "Set MODEL_ID explicitly or adjust MODEL_IMAGE_HINT." >&2
        exit 1
      fi
      echo "[info] Resolved MODEL_ID=${model_id} from MODEL_DISPLAY_NAME='${display_name}' and MODEL_IMAGE_HINT='${MODEL_IMAGE_HINT}'"
    else
      if [[ -n "$extra_filter" ]]; then
        filter="$extra_filter"
      fi
      model_id="$(resolve_model_id "$filter" "$MODEL_IMAGE_HINT" || true)"
      if [[ -z "${model_id// }" ]]; then
        echo "No matching custom model found in ${PROJECT_ID}/${REGION} with MODEL_IMAGE_HINT='${MODEL_IMAGE_HINT}'." >&2
        echo "Set MODEL_ID explicitly, set MODEL_DISPLAY_NAME, or adjust MODEL_IMAGE_HINT." >&2
        exit 1
      fi
      echo "[info] Resolved MODEL_ID=${model_id} from latest custom model (MODEL_IMAGE_HINT='${MODEL_IMAGE_HINT}')"
    fi
  fi

  echo "[action] Deploy model (auto-creates endpoint if missing)"
  if [[ -z "$ENDPOINT_ID" ]]; then
    PROJECT_ID="$PROJECT_ID" REGION="$REGION" MODEL_ID="$model_id" \
      "$ROOT_DIR/scripts/vertex_endpoint_control.sh" deploy-auto
  else
    PROJECT_ID="$PROJECT_ID" REGION="$REGION" VERTEX_ENDPOINT_ID="$ENDPOINT_ID" MODEL_ID="$model_id" \
      "$ROOT_DIR/scripts/vertex_endpoint_control.sh" deploy
  fi
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
