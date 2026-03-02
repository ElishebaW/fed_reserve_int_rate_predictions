from typing import Any, Callable


def generate_with_backoff(
    call_fn: Callable[[], Any],
    enforce_interval_fn: Callable[[], None],
    is_retryable_fn: Callable[[Exception], bool],
    sleep_fn: Callable[[float], None],
    jitter_fn: Callable[[], float],
    max_retries: int,
    base_seconds: float,
):
    attempts = 0
    while True:
        try:
            enforce_interval_fn()
            return call_fn()
        except Exception as exc:
            if attempts >= max_retries or not is_retryable_fn(exc):
                raise
            sleep_s = (base_seconds * (2 ** attempts)) + jitter_fn()
            sleep_fn(sleep_s)
            attempts += 1
