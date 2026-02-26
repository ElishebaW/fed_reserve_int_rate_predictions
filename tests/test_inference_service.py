from types import SimpleNamespace

import pytest
from google.api_core import exceptions as gax_exceptions

import inference_service as svc


class FakePredictionClient:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def endpoint_path(self, project: str, location: str, endpoint: str) -> str:
        return f"projects/{project}/locations/{location}/endpoints/{endpoint}"

    def predict(self, request, timeout):
        if self._error is not None:
            raise self._error
        return self._response


def test_enforce_feature_schema_passes_numeric_strings():
    required = ["Year", "Month", "Inflation Rate"]
    ranges = {"Year": (1900.0, 2100.0), "Month": (1.0, 12.0), "Inflation Rate": (-20.0, 40.0)}
    out = svc.enforce_feature_schema(
        {"Year": "2026", "Month": 3, "Inflation Rate": "2.7"},
        required,
        ranges,
    )
    assert out == {"Year": 2026.0, "Month": 3.0, "Inflation Rate": 2.7}


def test_enforce_feature_schema_rejects_missing_keys():
    with pytest.raises(svc.InferenceContractError, match="Missing required feature keys"):
        svc.enforce_feature_schema({"Year": 2026}, ["Year", "Month"])


def test_enforce_feature_schema_rejects_out_of_range():
    with pytest.raises(svc.InferenceContractError, match="out of range"):
        svc.enforce_feature_schema(
            {"Year": 2200, "Month": 1},
            ["Year", "Month"],
            {"Year": (1900.0, 2100.0), "Month": (1.0, 12.0)},
        )


def test_parse_prediction_value_supports_map_payloads():
    assert svc.parse_prediction_value({"prediction": "1.42"}) == 1.42
    assert svc.parse_prediction_value({"value": 0.5}) == 0.5


def test_vertex_service_predict_handles_timeout(monkeypatch):
    timeout_exc = gax_exceptions.DeadlineExceeded("deadline")
    fake_client = FakePredictionClient(error=timeout_exc)

    monkeypatch.setattr(svc.aiplatform.gapic, "PredictionServiceClient", lambda **_: fake_client)

    service = svc.VertexInferenceService(
        svc.VertexEndpointConfig(project_id="p", region="us-central1", endpoint_id="123", timeout_seconds=3)
    )

    with pytest.raises(svc.InferenceExecutionError, match="timed out"):
        service.predict({"Year": 2026.0})


def test_vertex_service_predict_parses_prediction(monkeypatch):
    fake_resp = SimpleNamespace(predictions=[{"prediction": 1.1}])
    fake_client = FakePredictionClient(response=fake_resp)
    monkeypatch.setattr(svc.aiplatform.gapic, "PredictionServiceClient", lambda **_: fake_client)

    service = svc.VertexInferenceService(
        svc.VertexEndpointConfig(project_id="p", region="us-central1", endpoint_id="123")
    )
    assert service.predict({"Year": 2026.0}) == 1.1
