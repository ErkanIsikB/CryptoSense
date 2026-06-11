import unittest
from unittest.mock import patch, MagicMock
import asyncio
import signal
from src.core.utils.signals import setup_signals

class TestSignals(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_setup_signals_sets_event(self):
        stop_event = asyncio.Event()
        
        # Test loop.add_signal_handler path or fallback.
        # Under Windows loop.add_signal_handler triggers NotImplementedError, causing fallback to signal.signal.
        with patch("asyncio.get_running_loop") as mock_get_loop, \
             patch("signal.signal") as mock_signal:
            
            mock_loop_instance = MagicMock()
            mock_loop_instance.add_signal_handler.side_effect = NotImplementedError
            mock_get_loop.return_value = mock_loop_instance
            
            handlers = {}
            def save_handler(sig, handler):
                handlers[sig] = handler
            mock_signal.side_effect = save_handler
            
            setup_signals(stop_event)
            
            self.assertIn(signal.SIGINT, handlers)
            self.assertIn(signal.SIGTERM, handlers)
            self.assertFalse(stop_event.is_set())
            
            # Execute handler manually to simulate signal catch
            handlers[signal.SIGINT]()
            self.assertTrue(stop_event.is_set())

if __name__ == "__main__":
    unittest.main()
