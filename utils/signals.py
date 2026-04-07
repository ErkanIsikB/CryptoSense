"""Shared signal handling for graceful shutdown."""

from __future__ import annotations

import asyncio
import signal


def setup_signals(stop: asyncio.Event) -> None:
    """Register SIGINT/SIGTERM handlers that set the *stop* event."""
    loop = asyncio.get_running_loop()

    def shutdown() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_: shutdown())
