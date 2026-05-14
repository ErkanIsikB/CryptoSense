import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db.db import execute_query_fetch
from src.models.lstm_autoencoder import LSTMAutoencoder

SYMBOL = "BTCUSDT"
SEQ_LEN = 12  # 12 buckets * 5 mins = 60 minutes of historical context


def fetch_training_data() -> pd.DataFrame:
    sql = """
          SELECT t.bucket, \
                 t.close, \
                 t.volume, \
                 t.vwap, \
                 o.avg_spread, \
                 o.avg_imbalance, \
                 COALESCE(s.avg_score, 0)    as avg_score, \
                 COALESCE(s.tweet_count, 0)  as tweet_count, \
                 COALESCE(c.net_flow_usd, 0) as net_flow_usd
          FROM trade_candles_5m t
                   LEFT JOIN orderbook_snapshots_5m o
                             ON t.bucket = o.bucket AND t.symbol = o.symbol
                   LEFT JOIN tweet_sentiment_5m s
                             ON t.bucket = s.bucket AND REPLACE(t.symbol, 'USDT', '') = s.symbol
                   LEFT JOIN (SELECT bucket, symbol, SUM(net_flow_usd) as net_flow_usd \
                              FROM cex_flows_5m \
                              GROUP BY bucket, symbol) c \
                             ON t.bucket = c.bucket AND REPLACE(t.symbol, 'USDT', '') = c.symbol
          WHERE t.symbol = %s
          ORDER BY t.bucket ASC \
          """
    rows = execute_query_fetch(sql, (SYMBOL,))

    columns = [
        'bucket', 'close', 'volume', 'vwap',
        'avg_spread', 'avg_imbalance', 'avg_score',
        'tweet_count', 'net_flow_usd'
    ]
    data = pd.DataFrame(rows, columns=columns)
    data.set_index('bucket', inplace=True)
    return data


print(f"Fetching historical data for {SYMBOL} from TimescaleDB...")
df = fetch_training_data()

if len(df) < SEQ_LEN:
    print(f"Not enough data in DB. Only found {len(df)} buckets. Keep the orchestrator running!")
    sys.exit()

# --- Feature Engineering for the Neural Net ---
# 1. Convert absolute price to % change
df['price_change_pct'] = df['close'].pct_change()

# 2. Calculate VWAP deviation (how far price is stretched from the volume average)
df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']

df.dropna(inplace=True)

# Select the final 8 features
features = df[[
    'price_change_pct', 'volume', 'vwap_dev',
    'avg_spread', 'avg_imbalance',
    'avg_score', 'tweet_count', 'net_flow_usd'
]].values

print(f"Usable Data Shape: {features.shape}")

# Scale Data using StandardScaler (better for financial outliers than MinMax)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Save the scaler so the live pipeline can use it
os.makedirs("scripts/data/anomalies", exist_ok=True)
with open("scripts/data/anomalies/anomaly_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f) # type: ignore

# Create overlapping sequences
sequences = []
for i in range(len(scaled_data) - SEQ_LEN):
    sequences.append(scaled_data[i:i + SEQ_LEN])

X_train = torch.tensor(np.array(sequences), dtype=torch.float32)

# Initialize LSTM with 8 features
model = LSTMAutoencoder(num_features=8, hidden_dim=64, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Training Autoencoder...")
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    reconstructed = model(X_train)
    loss = criterion(reconstructed, X_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Save the trained weights
torch.save(model.state_dict(), "scripts/data/anomalies/anomaly_model.pth")
print("Training complete. Model and Scaler successfully saved to scripts/data/anomalies/")