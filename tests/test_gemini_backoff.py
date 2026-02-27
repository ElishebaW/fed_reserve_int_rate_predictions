import pytest

from gemini_backoff import generate_with_backoff


def test_retryable_429_retries_then_succeeds():
    calls = {"n": 0}
    sleeps = []

    def call_fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("429 Resource exhausted")
        return "ok"

    out = generate_with_backoff(
        call_fn=call_fn,
        enforce_interval_fn=lambda: None,
        is_retryable_fn=lambda exc: "429" in str(exc),
        sleep_fn=lambda s: sleeps.append(s),
        jitter_fn=lambda: 0.0,
        max_retries=4,
        base_seconds=1.0,
    )

    assert out == "ok"
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_non_retryable_error_fails_immediately():
    calls = {"n": 0}

    def call_fn():
        calls["n"] += 1
        raise Exception("400 bad request")

    with pytest.raises(Exception, match="400 bad request"):
        generate_with_backoff(
            call_fn=call_fn,
            enforce_interval_fn=lambda: None,
            is_retryable_fn=lambda exc: False,
            sleep_fn=lambda s: None,
            jitter_fn=lambda: 0.0,
            max_retries=4,
            base_seconds=1.0,
        )

    assert calls["n"] == 1


def test_retryable_error_respects_max_retries():
    calls = {"n": 0}
    sleeps = []

    def call_fn():
        calls["n"] += 1
        raise Exception("429 quota")

    with pytest.raises(Exception, match="429 quota"):
        generate_with_backoff(
            call_fn=call_fn,
            enforce_interval_fn=lambda: None,
            is_retryable_fn=lambda exc: True,
            sleep_fn=lambda s: sleeps.append(s),
            jitter_fn=lambda: 0.0,
            max_retries=2,
            base_seconds=1.0,
        )

    # Initial call + 2 retries, then raise.
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]
