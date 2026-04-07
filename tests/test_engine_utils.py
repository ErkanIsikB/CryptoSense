import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bitquery_stream_engine import classify_transfer_flow


class TestEngineUtils(unittest.TestCase):
    def test_flow_inflow(self):
        self.assertEqual(classify_transfer_flow("Big Exchange", None), "inflow")

    def test_flow_outflow(self):
        self.assertEqual(classify_transfer_flow(None, "Centralized Exchange"), "outflow")

    def test_flow_neutral(self):
        self.assertEqual(classify_transfer_flow(None, None), "neutral")

if __name__ == "__main__":
    unittest.main()
