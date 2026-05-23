"""CEX Flow Ingestion — polls Bitquery every 5 minutes for exchange inflows/outflows.

Uses the Bitquery HTTP GraphQL endpoint with **server-side** CEX address filtering.
For each network, two queries are made:
  1. Inflows  — transfers where ``Receiver`` is a known CEX hot-wallet
  2. Outflows — transfers where ``Sender``   is a known CEX hot-wallet

Each poll cycle produces one aggregated row per (network, symbol, 5-min bucket)
written to ``cex_flows_5m``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from src.core.config import settings
from src.data_sources.bitquery.cex_addresses import (
    CEX_ADDRESSES_BY_NETWORK,
    TOKEN_CONTRACTS,
)
from src.db.db import execute_batch_async

LOGGER = logging.getLogger("cex_flow_ingestion")

BITQUERY_HTTP_URL = "https://streaming.bitquery.io/graphql"

INSERT_SQL = """
INSERT INTO cex_flows_5m
    (bucket, symbol, network, inflow_amount, inflow_usd,
     outflow_amount, outflow_usd, net_flow_usd,
     inflow_tx_count, outflow_tx_count)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (bucket, symbol, network) DO UPDATE SET
    inflow_amount    = EXCLUDED.inflow_amount,
    inflow_usd       = EXCLUDED.inflow_usd,
    outflow_amount   = EXCLUDED.outflow_amount,
    outflow_usd      = EXCLUDED.outflow_usd,
    net_flow_usd     = EXCLUDED.net_flow_usd,
    inflow_tx_count  = EXCLUDED.inflow_tx_count,
    outflow_tx_count = EXCLUDED.outflow_tx_count;
"""


# ── Query builders ──────────────────────────────────────────────


def _build_evm_cex_query(
    network: str, cex_addresses: list[str], direction: str
) -> str:
    """Build a Bitquery EVM query filtering by CEX address on sender or receiver.

    ``direction`` must be ``"inflow"`` or ``"outflow"``.
    - inflow:  Receiver is a CEX address (external → CEX)
    - outflow: Sender is a CEX address   (CEX → external)
    """
    addr_json = json.dumps(cex_addresses)

    if direction == "inflow":
        addr_filter = f'Receiver: {{in: {addr_json}}}'
    else:
        addr_filter = f'Sender: {{in: {addr_json}}}'

    return f"""
    {{
      EVM(network: {network}, dataset: realtime) {{
        Transfers(
          limit: {{count: 500}}
          orderBy: {{descending: Block_Time}}
          where: {{
            Block: {{Time: {{since_relative: {{minutes_ago: 10}}}}}}
            Transfer: {{
              AmountInUSD: {{gt: "100"}}
              {addr_filter}
            }}
          }}
        ) {{
          Transfer {{
            Amount
            AmountInUSD
            Sender
            Receiver
            Currency {{ Symbol SmartContract }}
          }}
          Block {{ Time }}
        }}
      }}
    }}
    """


def _build_solana_cex_query(cex_addresses: list[str], direction: str) -> str:
    """Build a Bitquery Solana query filtering by CEX address."""
    addr_json = json.dumps(cex_addresses)

    if direction == "inflow":
        addr_filter = f'Receiver: {{Address: {{in: {addr_json}}}}}'
    else:
        addr_filter = f'Sender: {{Address: {{in: {addr_json}}}}}'

    return f"""
    {{
      Solana(dataset: realtime) {{
        Transfers(
          limit: {{count: 500}}
          orderBy: {{descending: Block_Time}}
          where: {{
            Block: {{Time: {{since_relative: {{minutes_ago: 10}}}}}}
            Transfer: {{
              AmountInUSD: {{gt: "100"}}
              {addr_filter}
            }}
          }}
        ) {{
          Transfer {{
            Amount
            AmountInUSD
            Sender {{ Address }}
            Receiver {{ Address }}
            Currency {{ Symbol MintAddress }}
          }}
          Block {{ Time }}
        }}
      }}
    }}
    """


# ── Token mapping ──────────────────────────────────────────────

# Map token contract addresses to our tracked symbols
_TOKEN_SYMBOL_MAP: dict[str, str] = {}
for _net, _tokens in TOKEN_CONTRACTS.items():
    for _sym, _addr in _tokens.items():
        _TOKEN_SYMBOL_MAP[_addr.lower()] = _sym

# Also map common native token symbols
_NATIVE_SYMBOL_MAP: dict[str, str] = {
    "ETH": "ETH",
    "WETH": "ETH",
    "WBTC": "BTC",
    "BNB": "BNB",
    "WBNB": "BNB",
    "AVAX": "AVAX",
    "WAVAX": "AVAX",
    "SOL": "SOL",
    "WSOL": "SOL",
    "Wrapped SOL": "SOL",
    "USDT": "USDT",
    "USDC": "USDC",
}


def _resolve_symbol(transfer: dict[str, Any]) -> str | None:
    """Map a transfer's currency to one of our tracked symbols."""
    currency = transfer.get("Currency", {})

    # Try contract address first
    contract = (currency.get("SmartContract") or currency.get("MintAddress") or "").lower()
    if contract in _TOKEN_SYMBOL_MAP:
        return _TOKEN_SYMBOL_MAP[contract]

    # Try symbol name
    sym = currency.get("Symbol", "")
    return _NATIVE_SYMBOL_MAP.get(sym)


# ── Aggregation ────────────────────────────────────────────────


def _parse_block_time(tx: dict[str, Any]) -> float:
    """Extract block time from a Bitquery transaction and return Unix epoch seconds."""
    block = tx.get("Block", {})
    t_str = block.get("Time", "")
    if not t_str:
        return time.time()
    try:
        t_str = t_str.replace(" ", "T").replace("Z", "+00:00")
        if "+" not in t_str and t_str.count(":") == 2:
            t_str += "+00:00"
        dt = datetime.fromisoformat(t_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time()


def _aggregate_by_symbol(
    transfers: list[dict[str, Any]],
    cex_addresses: set[str],
    direction: str,
) -> dict[tuple[float, str], dict[str, float]]:
    """Aggregate transfers by (bucket_ts, symbol), classifying as inflow or outflow.

    Returns ``{(bucket_ts, symbol): {amount, usd, count}}``.
    """
    result: dict[tuple[float, str], dict[str, float]] = {}

    for tx in transfers:
        transfer = tx.get("Transfer", {})

        try:
            amount = float(transfer.get("Amount") or 0)
            amount_usd = float(transfer.get("AmountInUSD") or 0)
        except (ValueError, TypeError):
            continue

        if amount_usd < 100:
            continue

        symbol = _resolve_symbol(transfer)
        if symbol is None:
            continue

        # For inflows, skip if sender is also CEX (internal transfer)
        # For outflows, skip if receiver is also CEX (internal transfer)
        # Handle both EVM (flat string) and Solana ({Address: ...}) formats
        raw_sender = transfer.get("Sender", "")
        raw_receiver = transfer.get("Receiver", "")
        sender = (raw_sender.get("Address", "") if isinstance(raw_sender, dict) else str(raw_sender)).lower()
        receiver = (raw_receiver.get("Address", "") if isinstance(raw_receiver, dict) else str(raw_receiver)).lower()

        if direction == "inflow" and sender in cex_addresses:
            continue  # internal CEX-to-CEX
        if direction == "outflow" and receiver in cex_addresses:
            continue  # internal CEX-to-CEX

        # True Temporal Bucketing using parsed Block.Time
        tx_time = _parse_block_time(tx)
        bucket_ts = tx_time - (tx_time % settings.AGGREGATION_WINDOW_SECONDS)

        key = (bucket_ts, symbol)
        if key not in result:
            result[key] = {"amount": 0.0, "usd": 0.0, "count": 0}

        result[key]["amount"] += amount
        result[key]["usd"] += amount_usd
        result[key]["count"] += 1

    return result


# ── Network fetchers ───────────────────────────────────────────


async def _fetch_evm_direction(
    client: httpx.AsyncClient,
    network: str,
    cex_addresses: list[str],
    direction: str,
) -> dict[str, dict[str, float]]:
    """Fetch and aggregate EVM transfers for one direction."""
    query = _build_evm_cex_query(network, cex_addresses, direction)
    headers = {
        "Authorization": f"Bearer {settings.BITQUERY_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = await client.post(
            BITQUERY_HTTP_URL,
            json={"query": query},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        LOGGER.error(
            "bitquery EVM fetch failed: network=%s dir=%s error=%s",
            network, direction, exc,
        )
        return {}

    if "errors" in data:
        LOGGER.error(
            "bitquery EVM errors: network=%s dir=%s errors=%s",
            network, direction, data["errors"],
        )

    evm_data = (data.get("data") or {}).get("EVM", {})
    transfers = evm_data.get("Transfers", [])

    if not transfers:
        return {}

    cex_set = {a.lower() for a in cex_addresses}
    return _aggregate_by_symbol(transfers, cex_set, direction)


async def _fetch_solana_direction(
    client: httpx.AsyncClient,
    cex_addresses: list[str],
    direction: str,
) -> dict[str, dict[str, float]]:
    """Fetch and aggregate Solana transfers for one direction."""
    query = _build_solana_cex_query(cex_addresses, direction)
    headers = {
        "Authorization": f"Bearer {settings.BITQUERY_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = await client.post(
            BITQUERY_HTTP_URL,
            json={"query": query},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        LOGGER.error("bitquery Solana fetch failed: dir=%s error=%s", direction, exc)
        return {}

    if "errors" in data:
        LOGGER.error(
            "bitquery Solana errors: dir=%s errors=%s",
            direction, data["errors"],
        )

    solana_data = (data.get("data") or {}).get("Solana", {})
    transfers = solana_data.get("Transfers", [])

    if not transfers:
        return {}

    cex_set = {a for a in cex_addresses}  # Solana addresses are case-sensitive
    return _aggregate_by_symbol(transfers, cex_set, direction)


# ── Main poll cycle ────────────────────────────────────────────


async def _poll_once() -> None:
    """Run one CEX flow polling cycle across all configured networks.
    
    Implements sliding-window bucket finalization: queries the last 10 minutes
    of transaction history and commits ONLY the completed previous 5-minute
    bucket. This ensures no timing-drift gaps and zero partial entry overwrites.
    """
    now_epoch = time.time()
    # Target exactly the previous completed 5-minute bucket
    target_bucket_ts = now_epoch - (now_epoch % settings.AGGREGATION_WINDOW_SECONDS) - settings.AGGREGATION_WINDOW_SECONDS
    target_bucket_dt = datetime.fromtimestamp(target_bucket_ts, tz=timezone.utc)

    # Collect inflows and outflows per (bucket_ts, network, symbol)
    # Key: (bucket_ts, network, symbol) → {inflow_amount, inflow_usd, inflow_count,
    #                                     outflow_amount, outflow_usd, outflow_count}
    combined: dict[tuple[float, str, str], dict[str, float]] = {}

    async with httpx.AsyncClient(timeout=settings.CEX_FLOW_TIMEOUT_S) as client:
        tasks = []

        for network in settings.CEX_FLOW_NETWORKS:
            cex_addrs = CEX_ADDRESSES_BY_NETWORK.get(network, set())
            if not cex_addrs:
                continue

            cex_list = list(cex_addrs)

            if network == "solana":
                tasks.append(("solana", "inflow",
                    _fetch_solana_direction(client, cex_list, "inflow")))
                tasks.append(("solana", "outflow",
                    _fetch_solana_direction(client, cex_list, "outflow")))
            else:
                tasks.append((network, "inflow",
                    _fetch_evm_direction(client, network, cex_list, "inflow")))
                tasks.append((network, "outflow",
                    _fetch_evm_direction(client, network, cex_list, "outflow")))

        # Execute all queries concurrently
        coros = [t[2] for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for (network, direction, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                LOGGER.error("cex flow task failed: network=%s dir=%s err=%s",
                             network, direction, result)
                continue
            if not isinstance(result, dict):
                continue

            for (bucket_ts, symbol), agg in result.items():
                key = (bucket_ts, network, symbol)
                if key not in combined:
                    combined[key] = {
                        "inflow_amount": 0.0, "inflow_usd": 0.0, "inflow_count": 0,
                        "outflow_amount": 0.0, "outflow_usd": 0.0, "outflow_count": 0,
                    }

                if direction == "inflow":
                    combined[key]["inflow_amount"] += agg["amount"]
                    combined[key]["inflow_usd"] += agg["usd"]
                    combined[key]["inflow_count"] += agg["count"]
                else:
                    combined[key]["outflow_amount"] += agg["amount"]
                    combined[key]["outflow_usd"] += agg["usd"]
                    combined[key]["outflow_count"] += agg["count"]

    # Build DB rows ONLY for the completed target bucket
    rows: list[tuple[Any, ...]] = []
    for (bucket_ts, network, symbol), agg in combined.items():
        if bucket_ts != target_bucket_ts:
            continue  # Discard partial current and old buckets to avoid overwrite corruption
            
        if agg["inflow_count"] == 0 and agg["outflow_count"] == 0:
            continue
            
        net_flow = agg["inflow_usd"] - agg["outflow_usd"]
        rows.append((
            target_bucket_dt,
            symbol,
            network,
            agg["inflow_amount"],
            agg["inflow_usd"],
            agg["outflow_amount"],
            agg["outflow_usd"],
            net_flow,
            int(agg["inflow_count"]),
            int(agg["outflow_count"]),
        ))

    if rows:
        try:
            await execute_batch_async(INSERT_SQL, rows)
            LOGGER.info("wrote %d cex_flow row(s) to DB for finalized bucket %s", len(rows), target_bucket_dt)
            for r in rows:
                LOGGER.info(
                    "  cex_flow: %s/%s in=$%.0f(%d tx) out=$%.0f(%d tx) net=$%+.0f",
                    r[2], r[1], r[4], r[8], r[6], r[9], r[7],
                )
        except Exception:
            LOGGER.exception("failed to write cex_flows to DB")
    else:
        LOGGER.info("no CEX flow data finalized for bucket %s", target_bucket_dt)


async def start_cex_flow_stream(stop: asyncio.Event) -> None:
    """Public entry point — polls Bitquery for CEX flows every 5 minutes."""
    LOGGER.info("CEX flow ingestion started (poll every %ds)", settings.CEX_FLOW_POLL_INTERVAL_S)

    while not stop.is_set():
        try:
            await _poll_once()
        except Exception:
            LOGGER.exception("cex flow poll cycle failed")

        # Wait for next cycle or until stopped
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.CEX_FLOW_POLL_INTERVAL_S)
        except asyncio.TimeoutError:
            pass  # normal — poll again

    LOGGER.info("CEX flow ingestion stopped")
