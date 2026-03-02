from types import SimpleNamespace

import numpy as np
import pytest

import predictor


@pytest.fixture
def client(monkeypatch):
    predictor.feature_columns = [
        "Year",
        "Month",
        "Day",
        "Real GDP (Percent Change)",
        "Unemployment Rate",
        "Inflation Rate",
    ]
    predictor.model = SimpleNamespace(predict=lambda x: np.array([1.23] * len(x)))
    predictor.startup_error = None
    return predictor.app.test_client()


def test_parse_instances_success():
    predictor.feature_columns = ["Year", "Month"]
    frame = predictor.parse_instances({"instances": [{"Year": 2026, "Month": 2}]})
    assert list(frame.columns) == ["Year", "Month"]
    assert frame.iloc[0]["Year"] == 2026


def test_parse_instances_missing_column():
    predictor.feature_columns = ["Year", "Month"]
    with pytest.raises(predictor.RequestValidationError, match="Missing required feature columns"):
        predictor.parse_instances({"instances": [{"Year": 2026}]})


def test_parse_instances_unexpected_column():
    predictor.feature_columns = ["Year"]
    with pytest.raises(predictor.RequestValidationError, match="Unexpected feature columns"):
        predictor.parse_instances({"instances": [{"Year": 2026, "Foo": 1}]})


def test_predict_endpoint_validation_error(client):
    resp = client.post("/predict", json={"instances": [{"Year": 2026}]})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["code"] == "INVALID_INPUT"


def test_predict_endpoint_success(client):
    payload = {
        "instances": [
            {
                "Year": 2026,
                "Month": 2,
                "Day": 1,
                "Real GDP (Percent Change)": 2.2,
                "Unemployment Rate": 4.1,
                "Inflation Rate": 2.6,
            }
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert "predictions" in body
    assert isinstance(body["predictions"][0], float)


def test_predict_endpoint_model_not_ready(monkeypatch):
    predictor.model = None
    predictor.startup_error = "model missing"
    client = predictor.app.test_client()
    resp = client.post("/predict", json={"instances": []})
    assert resp.status_code == 500
    assert resp.get_json()["code"] == "MODEL_NOT_READY"
