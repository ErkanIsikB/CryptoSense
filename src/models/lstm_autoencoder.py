import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.decoder = LSTMDecoder(hidden_dim, input_dim, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)