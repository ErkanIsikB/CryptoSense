from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, TextIO

from sinks.base import BaseSink


class JsonlFileSink(BaseSink):
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._handles: dict[str, TextIO] = {}
        self._lock = asyncio.Lock()
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def write(self, key: str, record: dict[str, Any]) -> None:
        file_key = key.lower()
        serialized = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        async with self._lock:
            handle = self._get_handle(file_key)
            handle.write(serialized + "\n")
            handle.flush()

    async def close(self) -> None:
        async with self._lock:
            for handle in self._handles.values():
                handle.close()
            self._handles.clear()

    def _get_handle(self, key: str) -> TextIO:
        handle = self._handles.get(key)
        if handle is None:
            path = self._output_dir / f"{key}.jsonl"
            handle = path.open("a", encoding="utf-8", buffering=1)
            self._handles[key] = handle
        return handle
