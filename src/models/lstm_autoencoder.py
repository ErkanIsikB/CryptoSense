import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Reconstruct to original feature space
        self.output_layer = nn.Linear(hidden_dim, num_features)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)

        # Encode
        _, (hidden, cell) = self.encoder(x)

        # The hidden state from the last timestep is our context vector.
        # We repeat it for the sequence length to feed into the decoder.
        seq_len = x.shape[1]
        hidden_last = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # Decode
        decoder_out, _ = self.decoder(hidden_last)

        # Reconstruct
        reconstructed = self.output_layer(decoder_out)
        return reconstructed