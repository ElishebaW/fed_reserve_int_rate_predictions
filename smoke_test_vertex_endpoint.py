import json
import os
import sys
from pathlib import Path

from inference_service import VertexEndpointConfig, VertexInferenceService


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        os.environ.setdefault(key, value)


def main() -> int:
    load_env_file(Path(".env"))

    project_id = os.environ.get("PROJECT_ID", "").strip().strip('"')
    region = os.environ.get("REGION", "us-central1").strip().strip('"')
    endpoint_id = os.environ.get("VERTEX_ENDPOINT_ID", "").strip().strip('"')

    if not project_id:
        print("ERROR: PROJECT_ID is not set")
        return 2
    if not endpoint_id:
        print("ERROR: VERTEX_ENDPOINT_ID is not set")
        return 2

    instances_path = Path("instances.json")
    if not instances_path.exists():
        print("ERROR: instances.json not found")
        return 2

    payload = json.loads(instances_path.read_text(encoding="utf-8"))
    instances = payload.get("instances", [])
    if not instances:
        print("ERROR: instances.json has no instances")
        return 2

    features = instances[0]
    print("--- Vertex Endpoint Smoke Test ---")
    print(f"PROJECT_ID={project_id}")
    print(f"REGION={region}")
    print(f"ENDPOINT_ID={endpoint_id}")
    print(f"ENDPOINT_NAME=projects/{project_id}/locations/{region}/endpoints/{endpoint_id}")
    print("INPUT_INSTANCE=")
    print(json.dumps(features, indent=2))

    service = VertexInferenceService(
        VertexEndpointConfig(
            project_id=project_id,
            region=region,
            endpoint_id=endpoint_id,
            timeout_seconds=20.0,
        )
    )

    try:
        pred = service.predict(features)
        print(f"SUCCESS: prediction={pred}")
        return 0
    except Exception as exc:
        print(f"FAILURE: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
