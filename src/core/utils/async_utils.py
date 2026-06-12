import asyncio
import logging
from typing import Any, Coroutine

LOGGER = logging.getLogger("async_utils")

async def run_pipeline_loop(
    stop_event: asyncio.Event,
    queue: asyncio.Queue[Any],
    listener_coro: Coroutine[Any, Any, Any],
    processor_coro: Coroutine[Any, Any, Any],
    flusher_coro: Coroutine[Any, Any, Any] | None = None,
    close_sink_coro: Coroutine[Any, Any, Any] | None = None,
) -> None:
    """Coordinate execution, cancellation, and cleanup of ingestion pipelines with fail-fast monitoring."""
    flusher_task = asyncio.create_task(flusher_coro) if flusher_coro is not None else None
    listener_task = asyncio.create_task(listener_coro)
    processor_task = asyncio.create_task(processor_coro)

    stop_task = asyncio.create_task(stop_event.wait())

    try:
        # Monitor the stop event and core tasks; fail fast if listener or processor terminates early
        done, pending = await asyncio.wait(
            [stop_task, listener_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task not in done:
            for task in done:
                if task.exception() is not None:
                    LOGGER.error(f"Critical pipeline task crashed: {task.exception()}")
            stop_event.set()
    finally:
        if not stop_task.done():
            stop_task.cancel()

        listener_task.cancel()
        tasks_to_cancel = [listener_task]
        if flusher_task is not None:
            flusher_task.cancel()
            tasks_to_cancel.append(flusher_task)
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        await queue.join()
        processor_task.cancel()
        await asyncio.gather(processor_task, return_exceptions=True)
        if close_sink_coro is not None:
            await close_sink_coro
