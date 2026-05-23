"""Known CEX (Centralized Exchange) hot-wallet addresses per network.

These are well-known, publicly-documented deposit / hot-wallet addresses used
by major exchanges.  They are used by ``cex_flow_ingestion.py`` to classify
on-chain transfers as *inflow* (→ CEX) or *outflow* (CEX →).

Sources: Etherscan labels, Arkham Intelligence, Dune Analytics public dashboards.
"""

from __future__ import annotations

# ── Ethereum (ERC-20 / ETH) ────────────────────────────────────
ETHEREUM_CEX_ADDRESSES: set[str] = {
    # Binance
    "0x28c6c06298d514db089934071355e5743bf21d60",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",
    "0xf977814e90da44bfa03b6295a0616a897441acec",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3",
    "0x503828976d22510aad0201ac7ec88293211d23da",
    "0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740",
    "0x3cd751e6b0078be393132286c442345e68ff0aaa",
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2",
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13",
    "0xae2d4617c862309a3d75a0ffb358c7a5009c673f",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b",
    "0x236f9f97e0e62388479bf9e5ba4889e46b0273c3",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40",
}

# ── BSC (BEP-20 / BNB) ────────────────────────────────────────
BSC_CEX_ADDRESSES: set[str] = {
    # Binance (BNB hot wallets)
    "0x28c6c06298d514db089934071355e5743bf21d60",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",
    "0x8894e0a0c962cb723c1ef8a1b63d2b4f461c29f7",
    "0xe2fc31f816a9b94326492132018c3aecc4a93ae1",
    "0xf977814e90da44bfa03b6295a0616a897441acec",
}

# ── Avalanche (C-Chain) ───────────────────────────────────────
AVALANCHE_CEX_ADDRESSES: set[str] = {
    # Binance
    "0x9f8c163cba728e99993abe7495f06c0a3c8ac8b9",
    # Coinbase (Avalanche bridge)
    "0x503828976d22510aad0201ac7ec88293211d23da",
}

# ── Solana ────────────────────────────────────────────────────
SOLANA_CEX_ADDRESSES: set[str] = {
    # Binance
    "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9",
    "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
    # Coinbase
    "H8sMJSCQxfKiFTCfDR3DUMLPwcRbM61LGFJ8N4dK3WjS",
    "2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm",
    # Kraken
    "FWznbcNXWQuHTawe9RxvQ2LdCENssh12dsznf4RiouN5",
}

# ── Lookup by network name ────────────────────────────────────

CEX_ADDRESSES_BY_NETWORK: dict[str, set[str]] = {
    "ethereum": ETHEREUM_CEX_ADDRESSES,
    "eth": ETHEREUM_CEX_ADDRESSES,
    "bsc": BSC_CEX_ADDRESSES,
    "avalanche": AVALANCHE_CEX_ADDRESSES,
    "avax": AVALANCHE_CEX_ADDRESSES,
    "solana": SOLANA_CEX_ADDRESSES,
}

# Token contract addresses per network (to filter relevant transfers)
TOKEN_CONTRACTS: dict[str, dict[str, str]] = {
    "ethereum": {
        "ETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",   # WETH
        "BTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",   # WBTC
    },
    "bsc": {
        "BNB": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",   # WBNB
    },
    "avalanche": {
        "AVAX": "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",  # WAVAX
    },
    "avax": {
        "AVAX": "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",  # WAVAX
    },
    "solana": {
        "SOL": "So11111111111111111111111111111111111111112",
    },
}
