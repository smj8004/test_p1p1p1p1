from __future__ import annotations

from trader.data.binance import BinanceDataClient


def test_data_client_testnet_uses_futures_testnet_urls() -> None:
    client = BinanceDataClient(testnet=True)
    try:
        api = client.exchange.urls.get("api", {})
        assert isinstance(api, dict)
        assert str(api.get("fapiPublic", "")).startswith("https://testnet.binancefuture.com/")
        assert str(api.get("fapiPrivate", "")).startswith("https://testnet.binancefuture.com/")
    finally:
        client.close()
