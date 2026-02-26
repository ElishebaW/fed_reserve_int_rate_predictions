from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from google.api_core import exceptions as gax_exceptions
from google.cloud import aiplatform


class InferenceContractError(ValueError):
    """Raised when features violate the model contract."""


class InferenceExecutionError(RuntimeError):
    """Raised when Vertex execution fails."""


@dataclass(frozen=True)
class VertexEndpointConfig:
    project_id: str
    region: str
    endpoint_id: str
    timeout_seconds: float = 20.0


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception as exc:  # pragma: no cover
        raise InferenceContractError(f"Feature value is not numeric: {value}") from exc


def enforce_feature_schema(
    raw_features: Mapping[str, Any],
    required_features: Iterable[str],
    feature_ranges: Optional[Mapping[str, tuple[float, float]]] = None,
) -> Dict[str, float]:
    if not isinstance(raw_features, Mapping):
        raise InferenceContractError("Features payload must be an object.")

    required = list(required_features)
    raw_keys = set(raw_features.keys())
    required_keys = set(required)

    missing = sorted(required_keys - raw_keys)
    if missing:
        raise InferenceContractError(f"Missing required feature keys: {missing}")

    extra = sorted(raw_keys - required_keys)
    if extra:
        raise InferenceContractError(f"Unexpected feature keys: {extra}")

    normalized: Dict[str, float] = {}
    for name in required:
        value = raw_features.get(name)
        if value is None:
            raise InferenceContractError(f"Feature '{name}' cannot be null.")

        num_value = _as_float(value)
        if feature_ranges and name in feature_ranges:
            low, high = feature_ranges[name]
            if num_value < low or num_value > high:
                raise InferenceContractError(
                    f"Feature '{name}' out of range [{low}, {high}]."
                )
        normalized[name] = num_value

    return normalized


def parse_prediction_value(prediction: Any) -> float:
    if isinstance(prediction, (int, float)):
        return float(prediction)

    if isinstance(prediction, Mapping):
        for key in ("prediction", "value", "score"):
            if key in prediction:
                return _as_float(prediction[key])

    raise InferenceExecutionError(
        f"Unexpected Vertex prediction payload type: {type(prediction).__name__}"
    )


class VertexInferenceService:
    def __init__(self, config: VertexEndpointConfig):
        if not config.endpoint_id:
            raise InferenceExecutionError("VERTEX_ENDPOINT_ID is empty.")
        self.config = config
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options={"api_endpoint": f"{config.region}-aiplatform.googleapis.com"}
        )
        self.endpoint = (
            config.endpoint_id
            if config.endpoint_id.startswith("projects/")
            else self.client.endpoint_path(
                project=config.project_id,
                location=config.region,
                endpoint=config.endpoint_id,
            )
        )

    def predict(self, features: Mapping[str, Any]) -> float:
        try:
            response = self.client.predict(
                request={
                    "endpoint": self.endpoint,
                    "instances": [dict(features)],
                    "parameters": {},
                },
                timeout=self.config.timeout_seconds,
            )
        except gax_exceptions.DeadlineExceeded as exc:
            raise InferenceExecutionError(
                f"Vertex endpoint timed out after {self.config.timeout_seconds}s."
            ) from exc
        except gax_exceptions.GoogleAPICallError as exc:
            raise InferenceExecutionError(f"Vertex API call failed: {exc}") from exc

        if not response.predictions:
            raise InferenceExecutionError("Vertex endpoint returned zero predictions.")

        return parse_prediction_value(response.predictions[0])
