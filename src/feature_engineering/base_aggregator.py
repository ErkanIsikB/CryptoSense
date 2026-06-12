from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from src.db.db import execute_batch

_db_write_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="db_agg_writer")

class BaseTimeBucketAggregator:
    """Base class for thread-safe time-bucketed database aggregators."""

    def __init__(self, window_s: int, insert_sql: str, entity_name: str, logger: logging.Logger) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, int], Any] = {}
        self.window_s = window_s
        self.insert_sql = insert_sql
        self.entity_name = entity_name
        self.logger = logger

    def _bucket_start(self, time_ms: int) -> int:
        """Calculate the Unix epoch timestamp for the start of the current bucket."""
        return (time_ms // 1000) - ((time_ms // 1000) % self.window_s)

    def _should_flush(self, acc: Any) -> bool:
        """Determine if an accumulator has valid data that should be flushed.
        
        Subclasses should override this method.
        """
        raise NotImplementedError("Subclasses must implement _should_flush")

    def maybe_flush(self, current_time_ms: int) -> None:
        """Flush stale buckets (strictly older than the current bucket)."""
        now_bucket = self._bucket_start(current_time_ms)
        to_flush: list[tuple[Any, ...]] = []

        with self._lock:
            stale_keys = [k for k in self._buckets if k[1] < now_bucket]
            for key in stale_keys:
                acc = self._buckets.pop(key)
                if self._should_flush(acc):
                    to_flush.append(acc.to_row())

        self._write(to_flush)

    async def run_periodic_flusher(self, stop_event: asyncio.Event, interval_s: float = 15.0) -> None:
        """Periodically flush stale buckets until stop_event is set."""
        while not stop_event.is_set():
            try:
                await asyncio.sleep(interval_s)
                now_ms = int(time.time() * 1000)
                self.maybe_flush(now_ms)
            except Exception:
                self.logger.exception("error in periodic flusher for %s", self.entity_name)

    def flush_all(self) -> None:
        """Force-flush every open bucket (used at shutdown)."""
        with self._lock:
            rows = [acc.to_row() for acc in self._buckets.values() if self._should_flush(acc)]
            self._buckets.clear()
        self._write(rows, synchronous=True)

    def _write(self, rows: list[tuple[Any, ...]], synchronous: bool = False) -> None:
        """Write aggregated rows to the database, optionally in a background thread."""
        if not rows:
            return

        def run_in_background() -> None:
            try:
                execute_batch(self.insert_sql, rows)
                self.logger.info("flushed %d %s to DB", len(rows), self.entity_name)
            except Exception:
                self.logger.exception("failed to flush %s", self.entity_name)

        if synchronous:
            run_in_background()
        else:
            _db_write_executor.submit(run_in_background)
