from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny viewer for Bitquery raw_data JSONL files")
    parser.add_argument("--data-dir", default="raw_data", help="Directory containing JSON files")
    parser.add_argument("--date", help="Date prefix, e.g. 2026-03-31")
    parser.add_argument("--token", help="Token symbol, e.g. ETH")
    parser.add_argument("--category", choices=["transfers", "trades"], help="Data category")
    parser.add_argument("--limit", type=int, default=20, help="Max number of rows to display")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print full JSON objects")
    return parser.parse_args()


def pick_files(data_dir: Path, date: str | None, token: str | None, category: str | None) -> list[Path]:
    if not data_dir.exists():
        return []

    token_part = (token or "*").upper()
    category_part = category or "*"
    date_part = date or "*"
    pattern = f"{date_part}_{token_part}_{category_part}.json"
    return sorted(data_dir.glob(pattern))


def read_rows(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    parsed["_source_file"] = file_path.name
                    rows.append(parsed)
    rows.sort(key=lambda row: str(row.get("timestamp", "")))
    return rows


def summarize_row(row: dict[str, Any]) -> str:
    ts = row.get("timestamp", "-")
    token = row.get("token", "-")
    event_type = row.get("event_type", "-")
    amount_usd = row.get("amount_in_usd", "-")
    flow = row.get("flow_hint", "-")
    source = row.get("_source_file", "-")

    if event_type == "dex_trade":
        side = row.get("trade_side", "-")
        protocol = row.get("protocol_name", "-")
        price_usd = row.get("price_in_usd", "-")
        return (
            f"{ts} | {token} | trade | side={side} | protocol={protocol} | "
            f"amount_usd={amount_usd} | price_usd={price_usd} | flow={flow} | {source}"
        )

    sender = row.get("sender", "-")
    receiver = row.get("receiver", "-")
    amount = row.get("amount", "-")
    return (
        f"{ts} | {token} | transfer | amount={amount} | amount_usd={amount_usd} | "
        f"from={sender} | to={receiver} | flow={flow} | {source}"
    )


def main() -> None:
    args = parse_args()
    files = pick_files(Path(args.data_dir), args.date, args.token, args.category)

    if not files:
        print("No matching files found.")
        return

    rows = read_rows(files)
    if not rows:
        print("No readable JSON rows found.")
        return

    shown = rows[-max(args.limit, 1) :]
    print(f"Matched files: {len(files)} | Total rows: {len(rows)} | Showing: {len(shown)}")

    if args.pretty:
        for row in shown:
            print(json.dumps(row, ensure_ascii=False, indent=2))
    else:
        for row in shown:
            print(summarize_row(row))


if __name__ == "__main__":
    main()
