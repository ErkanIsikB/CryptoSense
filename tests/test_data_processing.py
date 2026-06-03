import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from src.models.retraining_service import slice_continuous_windows, align_labels_to_sequences

class TestDataProcessing(unittest.TestCase):
    def test_slice_continuous_windows(self):
        base_ts = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)
        timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(15)]
        
        feature_matrix = np.arange(15 * 3).reshape(15, 3).astype(np.float32)
        
        # Expect 15 - 12 + 1 = 4 sliding windows of length 12
        seqs = slice_continuous_windows(timestamps, feature_matrix, seq_len=12)
        self.assertEqual(seqs.shape, (4, 12, 3))
        
    def test_slice_continuous_windows_with_gaps(self):
        base_ts = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)
        # Create a gap: 6 elements, then a 50-min gap, then 6 elements
        timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(6)]
        gap_ts = base_ts + timedelta(minutes=50)
        timestamps += [gap_ts + timedelta(minutes=5 * i) for i in range(6)]
        
        feature_matrix = np.random.randn(12, 3).astype(np.float32)
        
        # Each block is length 6, which is < seq_len (12). So it should return 0 sequences.
        seqs = slice_continuous_windows(timestamps, feature_matrix, seq_len=12)
        self.assertEqual(len(seqs), 0)

    def test_align_labels_to_sequences(self):
        base_ts = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)
        timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(15)]
        df = pd.DataFrame({"bucket": timestamps})
        
        # Target label sequence (length 15)
        proxy_labels = np.zeros(15, dtype=np.int32)
        proxy_labels[11] = 1  # end of first window
        proxy_labels[14] = 1  # end of fourth window
        
        # Expected aligned labels length = 4 (for 4 sequences)
        aligned = align_labels_to_sequences(df, proxy_labels, seq_len=12)
        self.assertEqual(aligned.shape, (4,))
        self.assertEqual(aligned[0], 1)  # maps to proxy_labels[11]
        self.assertEqual(aligned[1], 0)  # maps to proxy_labels[12]
        self.assertEqual(aligned[2], 0)  # maps to proxy_labels[13]
        self.assertEqual(aligned[3], 1)  # maps to proxy_labels[14]

if __name__ == "__main__":
    unittest.main()
