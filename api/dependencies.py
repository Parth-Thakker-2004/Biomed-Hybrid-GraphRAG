"""Engine lifecycle for the API process."""

from __future__ import annotations

import threading

from core.pipeline import Engine

_engine: Engine | None = None
_lock = threading.Lock()

# Both models share one GPU; requests are served one at a time through this lock.
inference_lock = threading.Lock()


def get_engine() -> Engine:
    """Returns the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                engine = Engine()
                engine.warm_up()
                _engine = engine
    return _engine
