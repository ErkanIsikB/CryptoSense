"""Thread-safe registry for live anomaly models and scaler parameters."""

from __future__ import annotations

import threading
from typing import Any


class ModelRegistry:
    _models: dict[str, Any] = {}
    _scalers: dict[str, dict[str, Any]] = {}
    _locks: dict[str, threading.RLock] = {}
    _registry_lock = threading.RLock()

    @classmethod
    def _symbol_key(cls, symbol: str) -> str:
        return symbol.upper()

    @classmethod
    def _get_lock(cls, symbol: str) -> threading.RLock:
        key = cls._symbol_key(symbol)
        with cls._registry_lock:
            if key not in cls._locks:
                cls._locks[key] = threading.RLock()
            return cls._locks[key]

    @classmethod
    def register(cls, symbol: str, model: Any, scaler: dict[str, Any]) -> None:
        key = cls._symbol_key(symbol)
        lock = cls._get_lock(key)
        with lock:
            cls._models[key] = model
            cls._scalers[key] = scaler

    @classmethod
    def get(cls, symbol: str) -> tuple[Any | None, dict[str, Any] | None]:
        key = cls._symbol_key(symbol)
        lock = cls._get_lock(key)
        with lock:
            return cls._models.get(key), cls._scalers.get(key)

    @classmethod
    def hot_swap(cls, symbol: str, model: Any, scaler: dict[str, Any]) -> None:
        cls.register(symbol, model, scaler)

    @classmethod
    def registered_symbols(cls) -> tuple[str, ...]:
        with cls._registry_lock:
            return tuple(cls._models.keys())
