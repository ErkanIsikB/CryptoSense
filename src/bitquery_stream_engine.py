from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from websockets import connect
from websockets.exceptions import ConnectionClosed, InvalidStatusCode


LOGGER = logging.getLogger("bitquery_stream_engine")


@dataclass(frozen=True)
class SubscriptionSpec:
    name: str
    token: str
    chain: str
    category: str
    query: str
    variables: dict[str, Any]
    transformer: Callable[[dict[str, Any], str, str], dict[str, Any] | None]


EVM_WHALE_TRANSFERS_QUERY = """
subscription EvmWhaleTransfers($network: evm_network!, $minUsd: Float!, $since: DateTime!) {
  EVM(network: $network) {
    Transfers(
      where: {
        Block: { Time: { since: $since } }
        Transfer: { AmountInUSD: { gt: $minUsd } }
      }
    ) {
      Block { Time }
      Transfer {
        Amount
        AmountInUSD
        Sender
        Receiver
        SenderAnnotation
        ReceiverAnnotation
      }
      Currency { Symbol Name }
      Transaction { Hash }
    }
  }
}
""".strip()

EVM_DEX_TRADES_QUERY = """
subscription EvmDexTrades($network: evm_network!, $minUsd: Float!, $since: DateTime!) {
  EVM(network: $network) {
    DEXTrades(
      where: {
        Block: { Time: { since: $since } }
        Trade: { AmountInUSD: { gt: $minUsd } }
      }
    ) {
      Block { Time }
      Trade {
        Side
        AmountInUSD
        PriceInUSD
        Buyer
        Seller
      }
      Dex { ProtocolName ProtocolFamily }
      Buy { Currency { Symbol Name } }
      Sell { Currency { Symbol Name } }
      Transaction { Hash From To }
    }
  }
}
""".strip()

SOLANA_WHALE_TRANSFERS_QUERY = """
subscription SolanaWhaleTransfers($minUsd: Float!, $since: DateTime!) {
  Solana {
    Transfers(
      where: {
        Block: { Time: { since: $since } }
        Transfer: { AmountInUSD: { gt: $minUsd } }
      }
    ) {
      Block { Time }
      Transfer {
        Amount
        AmountInUSD
        Sender
        Receiver
        SenderAnnotation
        ReceiverAnnotation
      }
      Currency { Symbol Name }
      Transaction { Hash }
    }
  }
}
""".strip()

SOLANA_DEX_TRADES_QUERY = """
subscription SolanaDexTrades($minUsd: Float!, $since: DateTime!) {
  Solana {
    DEXTrades(
      where: {
        Block: { Time: { since: $since } }
        Trade: { AmountInUSD: { gt: $minUsd } }
      }
    ) {
      Block { Time }
      Trade {
        Side
        AmountInUSD
        PriceInUSD
        Buyer
        Seller
      }
      Dex { ProtocolName ProtocolFamily }
      Buy { Currency { Symbol Name } }
      Sell { Currency { Symbol Name } }
      Transaction { Hash }
    }
  }
}
""".strip()

BITCOIN_WHALE_TRANSFERS_QUERY = """
subscription BitcoinWhaleTransfers($minUsd: Float!, $since: DateTime!) {
  Bitcoin {
    Transfers(
      where: {
        Block: { Time: { since: $since } }
        Transfer: { AmountInUSD: { gt: $minUsd } }
      }
    ) {
      Block { Time }
      Transfer {
        Amount
        AmountInUSD
        Sender
        Receiver
        SenderAnnotation
        ReceiverAnnotation
      }
      Currency { Symbol Name }
      Transaction { Hash }
    }
  }
}
""".strip()


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


def build_output_file_path(root_dir: Path, date_str: str, token: str, category: str) -> Path:
    filename = f"{date_str}_{token}_{category}.json"
    return root_dir / filename


def parse_csv_values(raw: str | None, defaults: set[str]) -> set[str]:
    if raw is None:
        return defaults
    normalized = {part.strip().upper() for part in raw.split(",") if part.strip()}
    if not normalized:
        return defaults
    return normalized


def transform_transfer(row: dict[str, Any], chain: str, token: str) -> dict[str, Any] | None:
    event_time = deep_get(row, "Block.Time", "time")
    if not event_time:
        return None

    sender = deep_get(row, "Transfer.Sender", "Sender", "sender")
    receiver = deep_get(row, "Transfer.Receiver", "Receiver", "receiver")
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
        "currency_symbol": deep_get(row, "Currency.Symbol", "currency_symbol"),
        "tx_hash": deep_get(row, "Transaction.Hash", "tx_hash"),
        "raw": row,
    }


def transform_trade(row: dict[str, Any], chain: str, token: str) -> dict[str, Any] | None:
    event_time = deep_get(row, "Block.Time", "time")
    if not event_time:
        return None

    side = (deep_get(row, "Trade.Side", "Side", "side") or "").lower()
    flow_hint = "inflow" if side == "buy" else "outflow" if side == "sell" else "neutral"

    return {
        "event_type": "dex_trade",
        "chain": chain,
        "token": token,
        "timestamp": event_time,
        "trade_side": side or None,
        "protocol_name": deep_get(row, "Dex.ProtocolName", "ProtocolName", "protocol_name"),
        "protocol_family": deep_get(row, "Dex.ProtocolFamily", "ProtocolFamily", "protocol_family"),
        "amount_in_usd": as_float(deep_get(row, "Trade.AmountInUSD", "AmountInUSD", "amount_usd")),
        "price_in_usd": as_float(deep_get(row, "Trade.PriceInUSD", "PriceInUSD", "price_usd")),
        "buyer": deep_get(row, "Trade.Buyer", "Buyer", "buyer"),
        "seller": deep_get(row, "Trade.Seller", "Seller", "seller"),
        "buy_symbol": deep_get(row, "Buy.Currency.Symbol", "buy_symbol"),
        "sell_symbol": deep_get(row, "Sell.Currency.Symbol", "sell_symbol"),
        "flow_hint": flow_hint,
        "tx_hash": deep_get(row, "Transaction.Hash", "tx_hash"),
        "raw": row,
    }


class JsonSink:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, token: str, category: str, timestamp: str, record: dict[str, Any]) -> None:
        date_str = timestamp[:10]
        file_path = build_output_file_path(self.root_dir, date_str, token, category)
        serialized = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")


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
        sink: JsonSink,
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
        self.stop_event = asyncio.Event()

    def _since_iso(self) -> str:
        since = datetime.now(timezone.utc) - timedelta(minutes=self.backfill_minutes)
        return since.replace(microsecond=0).isoformat()

    def _subscription_specs(self) -> list[SubscriptionSpec]:
        since_iso = self._since_iso()
        specs = [
            SubscriptionSpec(
                name="eth_whales",
                token="ETH",
                chain="ethereum",
                category="transfers",
                query=EVM_WHALE_TRANSFERS_QUERY,
                variables={"network": "eth", "minUsd": self.whale_threshold, "since": since_iso},
                transformer=transform_transfer,
            ),
            SubscriptionSpec(
                name="bnb_whales",
                token="BNB",
                chain="bsc",
                category="transfers",
                query=EVM_WHALE_TRANSFERS_QUERY,
                variables={"network": "bsc", "minUsd": self.whale_threshold, "since": since_iso},
                transformer=transform_transfer,
            ),
            SubscriptionSpec(
                name="avax_whales",
                token="AVAX",
                chain="avalanche",
                category="transfers",
                query=EVM_WHALE_TRANSFERS_QUERY,
                variables={"network": "avalanche", "minUsd": self.whale_threshold, "since": since_iso},
                transformer=transform_transfer,
            ),
            SubscriptionSpec(
                name="eth_trades",
                token="ETH",
                chain="ethereum",
                category="trades",
                query=EVM_DEX_TRADES_QUERY,
                variables={"network": "eth", "minUsd": self.trade_min_usd, "since": since_iso},
                transformer=transform_trade,
            ),
            SubscriptionSpec(
                name="bnb_trades",
                token="BNB",
                chain="bsc",
                category="trades",
                query=EVM_DEX_TRADES_QUERY,
                variables={"network": "bsc", "minUsd": self.trade_min_usd, "since": since_iso},
                transformer=transform_trade,
            ),
            SubscriptionSpec(
                name="avax_trades",
                token="AVAX",
                chain="avalanche",
                category="trades",
                query=EVM_DEX_TRADES_QUERY,
                variables={"network": "avalanche", "minUsd": self.trade_min_usd, "since": since_iso},
                transformer=transform_trade,
            ),
            SubscriptionSpec(
                name="sol_whales",
                token="SOL",
                chain="solana",
                category="transfers",
                query=SOLANA_WHALE_TRANSFERS_QUERY,
                variables={"minUsd": self.whale_threshold, "since": since_iso},
                transformer=transform_transfer,
            ),
            SubscriptionSpec(
                name="sol_trades",
                token="SOL",
                chain="solana",
                category="trades",
                query=SOLANA_DEX_TRADES_QUERY,
                variables={"minUsd": self.trade_min_usd, "since": since_iso},
                transformer=transform_trade,
            ),
            SubscriptionSpec(
                name="btc_whales",
                token="BTC",
                chain="bitcoin",
                category="transfers",
                query=BITCOIN_WHALE_TRANSFERS_QUERY,
                variables={"minUsd": self.whale_threshold, "since": since_iso},
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
            except InvalidStatusCode as exc:
                wait_seconds = self._retry_delay(exc, backoff)
                LOGGER.warning("%s status=%s reconnecting in %ss", spec.name, exc.status_code, wait_seconds)
                await asyncio.sleep(wait_seconds)
                backoff = min(backoff * 2, 120)
            except Exception as exc:
                LOGGER.warning("%s disconnected (%s); reconnecting in %ss", spec.name, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 120)

    def _retry_delay(self, exc: InvalidStatusCode, default: int) -> int:
        if exc.status_code != 429:
            return default
        retry_after = None
        if hasattr(exc, "headers") and exc.headers is not None:
            retry_after = exc.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return max(int(retry_after), 5)
        return max(default, 30)

    async def _consume_stream(self, spec: SubscriptionSpec) -> None:
        LOGGER.info("connecting %s", spec.name)
        async with connect(
            self.stream_url,
            subprotocols=["graphql-transport-ws", "graphql-ws"],
            extra_headers={"X-API-KEY": self.api_key},
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        ) as ws:
            await ws.send(json.dumps({"type": "connection_init", "payload": {"headers": {"X-API-KEY": self.api_key}}}))
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
                if msg_type != "next":
                    continue

                payload_data = deep_get(message, "payload.data")
                if not isinstance(payload_data, dict):
                    continue

                rows = find_rows(payload_data)
                for row in rows:
                    record = spec.transformer(row, spec.chain, spec.token)
                    if record is None:
                        continue
                    timestamp = record.get("timestamp")
                    if isinstance(timestamp, str):
                        await self.sink.write(spec.token, spec.category, timestamp, record)

    def stop(self) -> None:
        self.stop_event.set()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_env(api_key: str) -> None:
    if not api_key:
        raise ValueError("BITQUERY_API_KEY is required")


def run() -> None:
    load_dotenv()

    api_key = os.getenv("BITQUERY_API_KEY", "").strip()
    stream_url = os.getenv("BITQUERY_STREAM_URL", "wss://streaming.bitquery.io/graphql").strip()
    whale_threshold = float(os.getenv("WHALE_USD_THRESHOLD", "500000"))
    trade_min_usd = float(os.getenv("TRADE_MIN_USD", "10000"))
    backfill_minutes = int(os.getenv("BACKFILL_MINUTES", "3"))
    raw_data_dir = Path(os.getenv("RAW_DATA_DIR", "./raw_data"))
    log_level = os.getenv("LOG_LEVEL", "INFO")
    enabled_tokens = parse_csv_values(
        os.getenv("ENABLED_TOKENS"),
        {"BTC", "ETH", "SOL", "BNB", "AVAX"},
    )
    enabled_categories = parse_csv_values(
        os.getenv("ENABLED_CATEGORIES"),
        {"TRANSFERS", "TRADES"},
    )
    enabled_categories = {value.lower() for value in enabled_categories}
    max_active_streams = int(os.getenv("MAX_ACTIVE_STREAMS", "0"))

    setup_logging(log_level)
    validate_env(api_key)

    sink = JsonSink(raw_data_dir)
    engine = StreamEngine(
        api_key=api_key,
        stream_url=stream_url,
        whale_threshold=whale_threshold,
        trade_min_usd=trade_min_usd,
        backfill_minutes=backfill_minutes,
        enabled_tokens=enabled_tokens,
        enabled_categories=enabled_categories,
        max_active_streams=max_active_streams,
        sink=sink,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, engine.stop)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(engine.run())
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
