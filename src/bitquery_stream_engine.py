"""Bitquery V2 Streaming API ingestion engine (whale transfers + DEX trades)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from websockets import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, InvalidStatusCode

from config import settings
from sinks.base import BaseSink
from sinks.jsonl_sink import JsonlFileSink

LOGGER = logging.getLogger("bitquery_stream_engine")


# ── Data types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SubscriptionSpec:
    name: str
    token: str
    chain: str
    category: str
    query: str
    variables: dict[str, Any]
    transformer: Callable[[dict[str, Any], str, str], dict[str, Any] | None]


# ── GraphQL subscription queries ───────────────────────────────────────

EVM_WHALE_TRANSFERS_QUERY = """
subscription EvmWhaleTransfers($network: evm_network!, $minUsd: String!) {
    EVM(network: $network, trigger_on: head) {
    Transfers(
      where: {
        Transfer: { AmountInUSD: { gt: $minUsd } }
      }
    ) {
      Block { Time }
      Transfer {
        Amount
        AmountInUSD
        Sender
        Receiver
                Currency { Symbol Name }
      }
      Transaction { Hash }
    }
  }
}
""".strip()

EVM_DEX_TRADES_QUERY = """
subscription EvmDexTrades($network: evm_network!, $minUsd: String!) {
    EVM(network: $network, trigger_on: head) {
    DEXTrades(
      where: {
                any: [
                    { Trade: { Buy: { AmountInUSD: { gt: $minUsd } } } },
                    { Trade: { Sell: { AmountInUSD: { gt: $minUsd } } } }
                ]
      }
    ) {
      Block { Time }
      Trade {
                Sender
                Dex { ProtocolName ProtocolFamily }
                Buy {
                    Amount
                    AmountInUSD
                    PriceInUSD
                    Buyer
                    Seller
                    Currency { Symbol Name }
                }
                Sell {
                    Amount
                    AmountInUSD
                    PriceInUSD
                    Buyer
                    Seller
                    Currency { Symbol Name }
                }
      }
      Transaction { Hash From To }
    }
  }
}
""".strip()

SOLANA_WHALE_TRANSFERS_QUERY = """
subscription SolanaWhaleTransfers($minUsd: String!) {
    Solana(trigger_on: head) {
    Transfers(
      where: {
        Transfer: { AmountInUSD: { gt: $minUsd } }
      }
    ) {
      Block { Time }
      Transfer {
        Amount
        AmountInUSD
                Sender { Address }
                Receiver { Address }
                Currency { Symbol Name }
      }
            Transaction { Signature }
    }
  }
}
""".strip()

SOLANA_DEX_TRADES_QUERY = """
subscription SolanaDexTrades($minUsd: String!) {
    Solana(trigger_on: head) {
    DEXTrades(
      where: {
                any: [
                    { Trade: { Buy: { AmountInUSD: { gt: $minUsd } } } },
                    { Trade: { Sell: { AmountInUSD: { gt: $minUsd } } } }
                ]
      }
    ) {
      Block { Time }
      Trade {
                Dex { ProtocolName ProtocolFamily }
                Buy {
                    Amount
                    AmountInUSD
                    PriceInUSD
                    Currency { Symbol Name }
                    Account { Address }
                }
                Sell {
                    Amount
                    AmountInUSD
                    PriceInUSD
                    Currency { Symbol Name }
                    Account { Address }
                }
      }
            Transaction { Signature }
    }
  }
}
""".strip()


# ── Utility helpers ─────────────────────────────────────────────────────


def deep_get(payload: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = payload
        ok = True
        for key in path.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                ok = False
                break
        if ok:
            return current
    return None


def find_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    stack: list[Any] = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, list) and node and isinstance(node[0], dict):
            return node
        if isinstance(node, dict):
            stack.extend(node.values())
    return []


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_transfer_flow(from_annotation: str | None, to_annotation: str | None) -> str:
    from_is_exchange = bool(from_annotation and "exchange" in from_annotation.lower())
    to_is_exchange = bool(to_annotation and "exchange" in to_annotation.lower())

    if from_is_exchange and not to_is_exchange:
        return "inflow"
    if to_is_exchange and not from_is_exchange:
        return "outflow"
    return "neutral"


# ── Transformers ────────────────────────────────────────────────────────


def transform_transfer(row: dict[str, Any], chain: str, token: str) -> dict[str, Any] | None:
    event_time = deep_get(row, "Block.Time", "time")
    if not event_time:
        return None

    sender = deep_get(row, "Transfer.Sender", "Transfer.Sender.Address", "Sender", "sender")
    receiver = deep_get(row, "Transfer.Receiver", "Transfer.Receiver.Address", "Receiver", "receiver")
    amount = as_float(deep_get(row, "Transfer.Amount", "Amount", "amount"))
    amount_usd = as_float(deep_get(row, "Transfer.AmountInUSD", "AmountInUSD", "amount_usd"))
    sender_annotation = deep_get(row, "Transfer.SenderAnnotation", "SenderAnnotation", "sender_annotation")
    receiver_annotation = deep_get(row, "Transfer.ReceiverAnnotation", "ReceiverAnnotation", "receiver_annotation")

    return {
        "event_type": "whale_transfer",
        "chain": chain,
        "token": token,
        "timestamp": event_time,
        "sender": sender,
        "receiver": receiver,
        "amount": amount,
        "amount_in_usd": amount_usd,
        "sender_annotation": sender_annotation,
        "receiver_annotation": receiver_annotation,
        "flow_hint": classify_transfer_flow(sender_annotation, receiver_annotation),
        "currency_symbol": deep_get(row, "Transfer.Currency.Symbol", "Currency.Symbol", "currency_symbol"),
        "tx_hash": deep_get(row, "Transaction.Hash", "Transaction.Signature", "tx_hash"),
        "raw": row,
    }


def transform_trade(row: dict[str, Any], chain: str, token: str) -> dict[str, Any] | None:
    event_time = deep_get(row, "Block.Time", "time")
    if not event_time:
        return None

    buy_symbol = deep_get(row, "Trade.Buy.Currency.Symbol", "Buy.Currency.Symbol", "buy_symbol")
    sell_symbol = deep_get(row, "Trade.Sell.Currency.Symbol", "Sell.Currency.Symbol", "sell_symbol")
    normalized_token = token.upper()
    if isinstance(buy_symbol, str) and buy_symbol.upper() == normalized_token:
        side = "buy"
    elif isinstance(sell_symbol, str) and sell_symbol.upper() == normalized_token:
        side = "sell"
    else:
        side = "neutral"

    flow_hint = "inflow" if side == "buy" else "outflow" if side == "sell" else "neutral"

    buy_amount_usd = as_float(deep_get(row, "Trade.Buy.AmountInUSD", "Buy.AmountInUSD"))
    sell_amount_usd = as_float(deep_get(row, "Trade.Sell.AmountInUSD", "Sell.AmountInUSD"))
    buy_price_usd = as_float(deep_get(row, "Trade.Buy.PriceInUSD", "Buy.PriceInUSD"))
    sell_price_usd = as_float(deep_get(row, "Trade.Sell.PriceInUSD", "Sell.PriceInUSD"))

    amount_in_usd = buy_amount_usd if side == "buy" else sell_amount_usd if side == "sell" else buy_amount_usd or sell_amount_usd
    price_in_usd = buy_price_usd if side == "buy" else sell_price_usd if side == "sell" else buy_price_usd or sell_price_usd

    return {
        "event_type": "dex_trade",
        "chain": chain,
        "token": token,
        "timestamp": event_time,
        "trade_side": side or None,
        "protocol_name": deep_get(row, "Trade.Dex.ProtocolName", "Dex.ProtocolName", "ProtocolName", "protocol_name"),
        "protocol_family": deep_get(row, "Trade.Dex.ProtocolFamily", "Dex.ProtocolFamily", "ProtocolFamily", "protocol_family"),
        "amount_in_usd": amount_in_usd,
        "price_in_usd": price_in_usd,
        "buyer": deep_get(
            row,
            "Trade.Buy.Buyer",
            "Trade.Sell.Buyer",
            "Trade.Buy.Account.Address",
            "Trade.Sell.Account.Address",
            "buyer",
        ),
        "seller": deep_get(
            row,
            "Trade.Buy.Seller",
            "Trade.Sell.Seller",
            "Trade.Sell.Account.Address",
            "Trade.Buy.Account.Address",
            "Trade.Sender",
            "seller",
        ),
        "buy_symbol": buy_symbol,
        "sell_symbol": sell_symbol,
        "flow_hint": flow_hint,
        "tx_hash": deep_get(row, "Transaction.Hash", "Transaction.Signature", "tx_hash"),
        "raw": row,
    }


# ── Stream engine ───────────────────────────────────────────────────────


class StreamEngine:
    def __init__(
        self,
        api_key: str,
        stream_url: str,
        whale_threshold: float,
        trade_min_usd: float,
        backfill_minutes: int,
        enabled_tokens: set[str],
        enabled_categories: set[str],
        max_active_streams: int,
        sink: BaseSink,
        stop_event: asyncio.Event,
    ) -> None:
        self.api_key = api_key
        self.stream_url = stream_url
        self.whale_threshold = whale_threshold
        self.trade_min_usd = trade_min_usd
        self.backfill_minutes = backfill_minutes
        self.enabled_tokens = enabled_tokens
        self.enabled_categories = enabled_categories
        self.max_active_streams = max_active_streams
        self.sink = sink
        self.stop_event = stop_event

    def _subscription_specs(self) -> list[SubscriptionSpec]:
        specs = [
            SubscriptionSpec(
                name="eth_whales",
                token="ETH",
                chain="ethereum",
                category="transfers",
                query=EVM_WHALE_TRANSFERS_QUERY,
                variables={"network": "eth", "minUsd": str(self.whale_threshold)},
                transformer=transform_transfer,
            ),
            SubscriptionSpec(
                name="bnb_whales",
                token="BNB",
                chain="bsc",
                category="transfers",
                query=EVM_WHALE_TRANSFERS_QUERY,
                variables={"network": "bsc", "minUsd": str(self.whale_threshold)},
                transformer=transform_transfer,
            ),
            SubscriptionSpec(
                name="avax_whales",
                token="AVAX",
                chain="avalanche",
                category="transfers",
                query=EVM_WHALE_TRANSFERS_QUERY,
                variables={"network": "avalanche", "minUsd": str(self.whale_threshold)},
                transformer=transform_transfer,
            ),
            SubscriptionSpec(
                name="eth_trades",
                token="ETH",
                chain="ethereum",
                category="trades",
                query=EVM_DEX_TRADES_QUERY,
                variables={"network": "eth", "minUsd": str(self.trade_min_usd)},
                transformer=transform_trade,
            ),
            SubscriptionSpec(
                name="bnb_trades",
                token="BNB",
                chain="bsc",
                category="trades",
                query=EVM_DEX_TRADES_QUERY,
                variables={"network": "bsc", "minUsd": str(self.trade_min_usd)},
                transformer=transform_trade,
            ),
            SubscriptionSpec(
                name="avax_trades",
                token="AVAX",
                chain="avalanche",
                category="trades",
                query=EVM_DEX_TRADES_QUERY,
                variables={"network": "avalanche", "minUsd": str(self.trade_min_usd)},
                transformer=transform_trade,
            ),
            SubscriptionSpec(
                name="sol_whales",
                token="SOL",
                chain="solana",
                category="transfers",
                query=SOLANA_WHALE_TRANSFERS_QUERY,
                variables={"minUsd": str(self.whale_threshold)},
                transformer=transform_transfer,
            ),
            SubscriptionSpec(
                name="sol_trades",
                token="SOL",
                chain="solana",
                category="trades",
                query=SOLANA_DEX_TRADES_QUERY,
                variables={"minUsd": str(self.trade_min_usd)},
                transformer=transform_transfer,
            ),
        ]

        filtered = [
            spec
            for spec in specs
            if spec.token in self.enabled_tokens and spec.category in self.enabled_categories
        ]

        if self.max_active_streams > 0:
            filtered = filtered[: self.max_active_streams]

        return filtered

    async def run(self) -> None:
        specs = self._subscription_specs()
        if not specs:
            LOGGER.warning(
                "No subscriptions enabled. Check ENABLED_TOKENS/ENABLED_CATEGORIES configuration."
            )
            return
        LOGGER.info("starting %s active subscriptions", len(specs))
        tasks = [asyncio.create_task(self._run_stream(spec)) for spec in specs]
        await self.stop_event.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_stream(self, spec: SubscriptionSpec) -> None:
        backoff = 1
        while not self.stop_event.is_set():
            try:
                await self._consume_stream(spec)
                backoff = 1
            except asyncio.CancelledError:
                raise
            except (InvalidStatusCode, InvalidStatus) as exc:
                wait_seconds = self._retry_delay(exc, backoff)
                status_code = getattr(exc, "status_code", None)
                if status_code is None and hasattr(exc, "response") and exc.response is not None:
                    status_code = getattr(exc.response, "status_code", None)
                LOGGER.warning("%s status=%s reconnecting in %ss", spec.name, status_code, wait_seconds)
                await asyncio.sleep(wait_seconds)
                backoff = min(backoff * 2, 120)
            except Exception as exc:
                LOGGER.warning("%s disconnected (%s); reconnecting in %ss", spec.name, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 120)

    def _retry_delay(self, exc: InvalidStatusCode | InvalidStatus, default: int) -> int:
        status_code = getattr(exc, "status_code", None)
        if status_code is None and hasattr(exc, "response") and exc.response is not None:
            status_code = getattr(exc.response, "status_code", None)
        if status_code != 429:
            return default
        retry_after = None
        headers = getattr(exc, "headers", None)
        if headers is None and hasattr(exc, "response") and exc.response is not None:
            headers = getattr(exc.response, "headers", None)
        if headers is not None:
            retry_after = headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return max(int(retry_after), 5)
        return max(default, 30)

    async def _consume_stream(self, spec: SubscriptionSpec) -> None:
        LOGGER.info("connecting %s", spec.name)
        async with connect(
            self.stream_url,
            subprotocols=["graphql-transport-ws", "graphql-ws"],
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-API-KEY": self.api_key,
            },
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "connection_init",
                        "payload": {
                            "headers": {
                                "Authorization": f"Bearer {self.api_key}",
                                "X-API-KEY": self.api_key,
                            }
                        },
                    }
                )
            )
            ack_raw = await ws.recv()
            ack = json.loads(ack_raw)
            if ack.get("type") not in {"connection_ack", "ka"}:
                raise RuntimeError(f"unexpected ack for {spec.name}: {ack}")

            sub_id = str(uuid.uuid4())
            await ws.send(
                json.dumps(
                    {
                        "id": sub_id,
                        "type": "subscribe",
                        "payload": {"query": spec.query, "variables": spec.variables},
                    }
                )
            )
            LOGGER.info("subscribed %s", spec.name)

            while not self.stop_event.is_set():
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=60)
                except asyncio.TimeoutError:
                    await ws.send(json.dumps({"type": "ping"}))
                    continue
                except ConnectionClosed as exc:
                    raise RuntimeError(f"connection closed for {spec.name}: {exc}") from exc

                message = json.loads(raw_msg)
                msg_type = message.get("type")

                if msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    continue
                if msg_type in {"pong", "ka"}:
                    continue
                if msg_type == "complete":
                    return
                if msg_type == "error":
                    raise RuntimeError(f"subscription error {spec.name}: {message}")
                if msg_type not in {"next", "data"}:
                    continue

                payload_data = deep_get(message, "payload.data")
                if not isinstance(payload_data, dict):
                    continue

                rows = find_rows(payload_data)
                for row in rows:
                    record = spec.transformer(row, spec.chain, spec.token)
                    if record is None:
                        continue
                    key = f"{spec.token}_{spec.category}"
                    await self.sink.write(key, record)

    def stop(self) -> None:
        self.stop_event.set()


# ── Public entry point ──────────────────────────────────────────────────


def _validate_env(api_key: str) -> None:
    if not api_key:
        raise ValueError("BITQUERY_API_KEY is required")


async def start_bitquery_stream(stop: asyncio.Event) -> None:
    """Public entry point — run the Bitquery streaming engine."""
    api_key = settings.BITQUERY_API_KEY
    _validate_env(api_key)

    sink = JsonlFileSink(settings.DATA_DIR / "bitquery")

    engine = StreamEngine(
        api_key=api_key,
        stream_url=settings.BITQUERY_STREAM_URL,
        whale_threshold=settings.WHALE_USD_THRESHOLD,
        trade_min_usd=settings.TRADE_MIN_USD,
        backfill_minutes=settings.BACKFILL_MINUTES,
        enabled_tokens=settings.ENABLED_TOKENS,
        enabled_categories=settings.ENABLED_CATEGORIES,
        max_active_streams=settings.MAX_ACTIVE_STREAMS,
        sink=sink,
        stop_event=stop,
    )

    try:
        await engine.run()
    finally:
        await sink.close()
