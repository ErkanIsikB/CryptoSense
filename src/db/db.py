"""Thread-safe connection pool for TimescaleDB (psycopg2).

Usage::

    from src.db import get_pool, execute_query, execute_batch, close_pool

    # Single row insert
    execute_query(
        "INSERT INTO my_table (a, b) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (val_a, val_b),
    )

    # Batch insert
    execute_batch(
        "INSERT INTO my_table (a, b) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        [(val_a1, val_b1), (val_a2, val_b2)],
    )
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Sequence
from psycopg2 import pool as pg_pool
from psycopg2.extras import execute_batch as _pg_execute_batch

from src.core.config import settings

LOGGER = logging.getLogger("db")

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()

MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 10


def get_pool() -> pg_pool.ThreadedConnectionPool:
    """Return (and lazily create) the global connection pool."""
    global _pool
    if _pool is not None and not _pool.closed:
        return _pool

    with _pool_lock:
        if _pool is not None and not _pool.closed:
            return _pool

        LOGGER.info("creating TimescaleDB connection pool")
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=MIN_CONNECTIONS,
            maxconn=MAX_CONNECTIONS,
            dsn=settings.DB_URL,
        )
        return _pool


@contextmanager
def get_connection() -> Iterator[Any]:
    """Borrow a connection from the pool. Discards broken connections dynamically."""
    p = get_pool()
    conn = p.getconn()
    close_conn = False
    try:
        yield conn
    except Exception:
        if conn.closed or getattr(conn, "broken", False):
            close_conn = True
        raise
    finally:
        try:
            p.putconn(conn, close=close_conn)
        except Exception:
            LOGGER.exception("Failed returning connection to psycopg2 pool")


def execute_query(
    sql: str,
    params: tuple[Any, ...] | None = None,
    *,
    commit: bool = True,
) -> None:
    """Execute a single SQL statement."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        if commit:
            conn.commit()


def execute_batch(
    sql: str,
    params_seq: Sequence[tuple[Any, ...]],
    *,
    page_size: int = 100,
    commit: bool = True,
) -> None:
    """Execute a parameterised SQL statement for a batch of rows."""
    if not params_seq:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            _pg_execute_batch(cur, sql, params_seq, page_size=page_size)
        if commit:
            conn.commit()


def execute_query_fetch(
    sql: str,
    params: tuple[Any, ...] | None = None,
) -> list[tuple[Any, ...]]:
    """Execute a SELECT and return all rows."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_migration(sql_path: str | None = None) -> None:
    """Execute the schema migration SQL file."""
    from pathlib import Path

    if sql_path is None:
        sql_path = str(Path(__file__).parent / "db_schema.sql")

    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    LOGGER.info("running schema migration from %s", sql_path)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    LOGGER.info("schema migration completed successfully")


def close_pool() -> None:
    """Shut down the connection pool."""
    global _pool
    with _pool_lock:
        if _pool is not None and not _pool.closed:
            _pool.closeall()
            LOGGER.info("TimescaleDB connection pool closed")
            _pool = None


# ── Asynchronous DB Wrappers (ThreadPool offloading + Semaphores) ────
import asyncio

_async_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _async_semaphore
    if _async_semaphore is None:
        _async_semaphore = asyncio.Semaphore(MAX_CONNECTIONS)
    return _async_semaphore


async def execute_query_async(
    sql: str,
    params: tuple[Any, ...] | None = None,
    *,
    commit: bool = True,
) -> None:
    """Execute a single SQL statement asynchronously in a thread-pool worker."""
    async with _get_semaphore():
        await asyncio.to_thread(execute_query, sql, params, commit=commit)


async def execute_batch_async(
    sql: str,
    params_seq: Sequence[tuple[Any, ...]],
    *,
    page_size: int = 100,
    commit: bool = True,
) -> None:
    """Execute a parameterised SQL statement for a batch of rows asynchronously."""
    if not params_seq:
        return
    async with _get_semaphore():
        await asyncio.to_thread(execute_batch, sql, params_seq, page_size=page_size, commit=commit)


async def execute_query_fetch_async(
    sql: str,
    params: tuple[Any, ...] | None = None,
) -> list[tuple[Any, ...]]:
    """Execute a SELECT query asynchronously and return all fetched rows."""
    async with _get_semaphore():
        return await asyncio.to_thread(execute_query_fetch, sql, params)
