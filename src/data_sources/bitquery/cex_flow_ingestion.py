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
from src.db.db import execute_batch

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
            Block: {{Time: {{since_relative: {{minutes_ago: 5}}}}}}
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
            Block: {{Time: {{since_relative: {{minutes_ago: 5}}}}}}
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


def _aggregate_by_symbol(
    transfers: list[dict[str, Any]],
    cex_addresses: set[str],
    direction: str,
) -> dict[str, dict[str, float]]:
    """Aggregate transfers by symbol, classifying as inflow or outflow.

    Returns ``{symbol: {amount, usd, count}}``.
    """
    result: dict[str, dict[str, float]] = {}

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

        if symbol not in result:
            result[symbol] = {"amount": 0.0, "usd": 0.0, "count": 0}

        result[symbol]["amount"] += amount
        result[symbol]["usd"] += amount_usd
        result[symbol]["count"] += 1

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
    """Run one CEX flow polling cycle across all configured networks."""
    now_epoch = time.time()
    bucket_start = now_epoch - (now_epoch % settings.AGGREGATION_WINDOW_SECONDS)
    bucket = datetime.fromtimestamp(bucket_start, tz=timezone.utc)

    # Collect inflows and outflows per (network, symbol)
    # Key: (network, symbol) → {inflow_amount, inflow_usd, inflow_count,
    #                            outflow_amount, outflow_usd, outflow_count}
    combined: dict[tuple[str, str], dict[str, float]] = {}

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

            for symbol, agg in result.items():
                key = (network, symbol)
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

    # Build DB rows
    rows: list[tuple[Any, ...]] = []
    for (network, symbol), agg in combined.items():
        if agg["inflow_count"] == 0 and agg["outflow_count"] == 0:
            continue
        net_flow = agg["inflow_usd"] - agg["outflow_usd"]
        rows.append((
            bucket,
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
            execute_batch(INSERT_SQL, rows)
            LOGGER.info("wrote %d cex_flow row(s) to DB", len(rows))
            for r in rows:
                LOGGER.info(
                    "  cex_flow: %s/%s in=$%.0f(%d tx) out=$%.0f(%d tx) net=$%+.0f",
                    r[2], r[1], r[4], r[8], r[6], r[9], r[7],
                )
        except Exception:
            LOGGER.exception("failed to write cex_flows to DB")
    else:
        LOGGER.info("no CEX flow data in this 5-min cycle")


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
