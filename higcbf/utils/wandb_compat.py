from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any


class _DummyConfig(dict):
    def update(self, obj: Any = None, **kwargs):
        if obj is not None:
            if isinstance(obj, Mapping):
                super().update(obj)
            elif isinstance(obj, SimpleNamespace):
                super().update(vars(obj))
            elif hasattr(obj, "__dict__"):
                super().update(vars(obj))
        if len(kwargs) > 0:
            super().update(kwargs)


class _DummyWandb:
    def __init__(self):
        self.config = _DummyConfig()

    def login(self, *args, **kwargs):
        return None

    def init(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None


try:
    import wandb as _wandb  # type: ignore
    wandb = _wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = _DummyWandb()

