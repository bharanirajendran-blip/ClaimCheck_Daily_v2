"""
Shared utilities: retry decorator, logging setup, env helpers.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def retry(times: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """Exponential back-off retry decorator."""
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt == times:
                        logger.error("%s failed after %d attempts: %s", fn.__name__, times, exc)
                        raise
                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.1fs…",
                        fn.__name__, attempt, times, exc, wait,
                    )
                    time.sleep(wait)
                    wait *= backoff
        return wrapper  # type: ignore[return-value]
    return decorator


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def require_env(*keys: str) -> dict[str, str]:
    """Return a dict of env vars, raising if any are missing."""
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example → .env and fill in the values."
        )
    return {k: os.environ[k] for k in keys}
