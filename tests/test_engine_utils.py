import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bitquery_stream_engine import build_output_file_path, classify_transfer_flow, parse_csv_values


class TestEngineUtils(unittest.TestCase):
    def test_flow_inflow(self):
        self.assertEqual(classify_transfer_flow("Big Exchange", None), "inflow")

    def test_flow_outflow(self):
        self.assertEqual(classify_transfer_flow(None, "Centralized Exchange"), "outflow")

    def test_flow_neutral(self):
        self.assertEqual(classify_transfer_flow(None, None), "neutral")

    def test_output_file_path(self):
        root = Path("raw_data")
        got = build_output_file_path(root, "2026-03-31", "ETH", "trades")
        self.assertEqual(got.as_posix(), "raw_data/2026-03-31_ETH_trades.json")

    def test_parse_csv_values_uses_defaults_on_none(self):
        defaults = {"ETH", "SOL"}
        self.assertEqual(parse_csv_values(None, defaults), defaults)

    def test_parse_csv_values_normalizes_values(self):
        defaults = {"ETH"}
        got = parse_csv_values(" btc, sol ,ETH ", defaults)
        self.assertEqual(got, {"BTC", "SOL", "ETH"})


if __name__ == "__main__":
    unittest.main()
