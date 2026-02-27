import json
from pathlib import Path

from prompt_sequence_harness import infer_from_prompt_sequence, run_prompt_sequence


PROMPTS = [
    "What will be the fed rate in 2028 with 10% unemployment",
    "June, 10, $24.11 trillion, 5%",
]


def load_required_features():
    schema_path = Path("model/feature_columns.json")
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    return payload["feature_columns"]


def test_prompt_sequence_fills_all_required_features():
    required_features = load_required_features()
    features, missing = run_prompt_sequence(PROMPTS, required_features)

    assert missing == []
    assert features["Year"] == 2028.0
    assert features["Month"] == 6.0
    assert features["Day"] == 10.0
    assert features["Real GDP (Percent Change)"] == 24.11
    assert features["Unemployment Rate"] == 10.0
    assert features["Inflation Rate"] == 5.0


def test_prompt_sequence_invokes_predictor_only_when_complete():
    required_features = load_required_features()
    captured = {}

    def fake_predictor(features):
        captured["features"] = features
        return 3.125

    pred = infer_from_prompt_sequence(PROMPTS, required_features, fake_predictor)

    assert pred == 3.125
    assert captured["features"]["Year"] == 2028.0
    assert captured["features"]["Real GDP (Percent Change)"] == 24.11
