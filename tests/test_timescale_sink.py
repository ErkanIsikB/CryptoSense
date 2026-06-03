import unittest
from unittest.mock import patch, MagicMock
import asyncio
from src.sinks.timescale_sink import TimescaleSink

class TestTimescaleSink(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("src.sinks.timescale_sink.TradeAggregator")
    @patch("src.sinks.timescale_sink.OrderbookAggregator")
    @patch("src.sinks.timescale_sink.score_and_store")
    def test_write_routing(self, mock_score_and_store, mock_orderbook_class, mock_trade_class):
        async def run_test():
            # Set up mock aggregators
            mock_trade_agg = MagicMock()
            mock_orderbook_agg = MagicMock()
            mock_trade_class.return_value = mock_trade_agg
            mock_orderbook_class.return_value = mock_orderbook_agg
            
            sink = TimescaleSink()
            
            # 1. Test Trade Record routing
            trade_record = {
                "type": "aggTrade",
                "symbol": "BTCUSDT",
                "price": 50000.0,
                "qty": 1.5,
                "trade_time_ms": 1700000000000,
                "is_buyer_maker": False
            }
            await sink.write("BTCUSDT", trade_record)
            mock_trade_agg.add.assert_called_once_with(
                symbol="BTCUSDT",
                price=50000.0,
                qty=1.5,
                trade_time_ms=1700000000000,
                is_buyer_maker=False
            )
            
            # 2. Test Orderbook Record routing
            ob_record = {
                "type": "depth_snapshot",
                "symbol": "ETHUSDT",
                "event_time_ms": 1700000010000,
                "bids": [["2000.0", "10.0"]],
                "asks": [["2001.0", "5.0"]]
            }
            await sink.write("ETHUSDT", ob_record)
            mock_orderbook_agg.add.assert_called_once_with(
                symbol="ETHUSDT",
                event_time_ms=1700000010000,
                bids=[["2000.0", "10.0"]],
                asks=[["2001.0", "5.0"]]
            )
            
            # 3. Test Sentiment Record routing
            sentiment_record = {
                "event_type": "sentiment",
                "symbol": "BTCUSDT",
                "text": "Bullish news"
            }
            await sink.write("BTCUSDT", sentiment_record)
            mock_score_and_store.assert_called_once_with(sentiment_record)
            
            # 4. Close flushes both aggregators
            await sink.close()
            mock_trade_agg.flush_all.assert_called_once()
            mock_orderbook_agg.flush_all.assert_called_once()

        self.loop.run_until_complete(run_test())

if __name__ == "__main__":
    unittest.main()
