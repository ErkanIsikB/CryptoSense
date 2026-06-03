import unittest
from unittest.mock import patch, MagicMock
import asyncio
from src.core.utils.retraining_scheduler import (
    start_retraining_scheduler,
    shutdown_retraining_scheduler,
    retrain_job
)

class TestRetrainingScheduler(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()
        shutdown_retraining_scheduler()

    @patch("src.core.utils.retraining_scheduler.AsyncIOScheduler")
    def test_start_and_shutdown_scheduler(self, mock_scheduler_class):
        mock_sched = MagicMock()
        mock_sched.running = False
        mock_scheduler_class.return_value = mock_sched
        
        # Start
        start_retraining_scheduler()
        
        # Verify AsyncIOScheduler instantiated and started
        mock_scheduler_class.assert_called_once()
        mock_sched.add_job.assert_called_once()
        mock_sched.start.assert_called_once()
        
        # Shutdown
        mock_sched.running = True
        shutdown_retraining_scheduler()
        mock_sched.shutdown.assert_called_once_with(wait=False)

    @patch("src.core.utils.retraining_scheduler.train_symbol_model")
    @patch("src.core.utils.retraining_scheduler.settings")
    def test_retrain_job(self, mock_settings, mock_train_symbol):
        async def run_test():
            mock_settings.RETRAIN_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
            mock_settings.RETRAIN_OUTPUT_DIR = "mock_out"
            mock_settings.RETRAIN_LOOKBACK_DAYS = 14
            
            # Mock return value of train_symbol_model
            mock_artifacts = MagicMock()
            mock_artifacts.version_dir = "mock_dir"
            mock_train_symbol.return_value = mock_artifacts
            
            # Execute retrain_job
            await retrain_job()
            
            # Verify training was triggered for both symbols
            self.assertEqual(mock_train_symbol.call_count, 2)
            mock_train_symbol.assert_any_call(
                "BTCUSDT",
                output_root="mock_out",
                lookback_days=14,
                hot_swap=True
            )
            mock_train_symbol.assert_any_call(
                "ETHUSDT",
                output_root="mock_out",
                lookback_days=14,
                hot_swap=True
            )

        self.loop.run_until_complete(run_test())

if __name__ == "__main__":
    unittest.main()
