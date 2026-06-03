import unittest
import torch
from src.models.lstm_autoencoder import LSTMAutoencoder

class TestLSTMAutoencoder(unittest.TestCase):
    def test_dimensions_and_forward_pass(self):
        batch_size = 4
        seq_len = 12
        input_dim = 19
        hidden_dim = 10
        
        model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len)
        
        # Create mock input tensor (batch, seq_len, features)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        reconstructed = model(x)
        
        # Assertions
        self.assertEqual(reconstructed.shape, x.shape)
        self.assertFalse(torch.isnan(reconstructed).any())
        
    def test_parameters_gradients(self):
        model = LSTMAutoencoder(input_dim=5, hidden_dim=3, seq_len=10)
        x = torch.randn(2, 10, 5)
        out = model(x)
        loss = torch.mean((out - x) ** 2)
        loss.backward()
        
        # Assert parameters have gradients
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")

if __name__ == "__main__":
    unittest.main()
