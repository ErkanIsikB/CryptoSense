"""APScheduler lifecycle helpers for periodic anomaly model retraining."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
except ImportError:  # pragma: no cover - exercised only before dependencies install
    AsyncIOScheduler = None

from src.core.config import settings
from src.models.retraining_service import train_symbol_model

LOGGER = logging.getLogger("retraining_scheduler")

_scheduler: Any | None = None
_retrain_lock = asyncio.Lock()


async def retrain_job() -> None:
    if _retrain_lock.locked():
        LOGGER.warning("Retraining job already running; skipping overlapping trigger")
        return

    async with _retrain_lock:
        for symbol in settings.RETRAIN_SYMBOLS:
            try:
                LOGGER.info("Starting scheduled retraining for %s", symbol)
                artifacts = await asyncio.to_thread(
                    train_symbol_model,
                    symbol,
                    output_root=settings.RETRAIN_OUTPUT_DIR,
                    lookback_days=settings.RETRAIN_LOOKBACK_DAYS,
                    hot_swap=True,
                )
                if artifacts is None:
                    LOGGER.warning("Scheduled retraining produced no artifacts for %s", symbol)
                    continue
                LOGGER.info("Completed scheduled retraining for %s at %s", symbol, artifacts.version_dir)
            except Exception:
                LOGGER.exception("Scheduled retraining failed for %s", symbol)


def start_retraining_scheduler() -> Any:
    global _scheduler

    if AsyncIOScheduler is None:
        raise RuntimeError("APScheduler is required for scheduled retraining. Install apscheduler.")

    if _scheduler is not None and _scheduler.running:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone=settings.RETRAIN_TIMEZONE)
    _scheduler.add_job(
        retrain_job,
        trigger="interval",
        days=settings.RETRAIN_INTERVAL_DAYS,
        id="retrain_job",
        replace_existing=True,
        misfire_grace_time=settings.RETRAIN_MISFIRE_GRACE_SECONDS,
        coalesce=True,
        max_instances=1,
    )
    _scheduler.start()
    LOGGER.info(
        "Retraining scheduler started: every %d day(s), symbols=%s",
        settings.RETRAIN_INTERVAL_DAYS,
        ",".join(settings.RETRAIN_SYMBOLS),
    )
    return _scheduler


def shutdown_retraining_scheduler() -> None:
    global _scheduler

    if _scheduler is None:
        return

    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        LOGGER.info("Retraining scheduler stopped")
    _scheduler = None
