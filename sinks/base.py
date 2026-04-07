from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSink(ABC):
    @abstractmethod
    async def write(self, key: str, record: dict[str, Any]) -> None:
        """Persist one record under a logical key."""

    @abstractmethod
    async def close(self) -> None:
        """Release sink resources."""
