"""Web control plane for batch trajectory annotation.

``fastapi`` and ``uvicorn`` are optional extras, so :func:`create_app` is imported
lazily — importing :mod:`longvideo_polar` must keep working on machines that only
run rollouts.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def create_app(**kwargs: Any) -> Any:
    from longvideo_polar.webui.server import create_app as _create_app

    return _create_app(**kwargs)
