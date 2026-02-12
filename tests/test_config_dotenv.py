from __future__ import annotations

from pathlib import Path

from trader.config import AppConfig


def test_from_env_loads_dotenv_file(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "BINANCE_API_KEY=abc123",
                "BINANCE_API_SECRET='secret456'",
                "BINANCE_ENV=mainnet",
                "LIVE_TRADING=true",
            ]
        ),
        encoding="utf-8",
    )
    cfg = AppConfig.from_env()
    assert cfg.binance_api_key is not None
    assert cfg.binance_api_secret is not None
    assert cfg.binance_api_key.get_secret_value() == "abc123"
    assert cfg.binance_api_secret.get_secret_value() == "secret456"
    assert cfg.binance_env == "mainnet"
    assert cfg.live_trading is True


def test_from_env_uses_dotenv_example_as_fallback(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "BINANCE_API_KEY=fallback_key",
                "BINANCE_API_SECRET=fallback_secret",
            ]
        ),
        encoding="utf-8",
    )
    cfg = AppConfig.from_env()
    assert cfg.binance_api_key is not None
    assert cfg.binance_api_secret is not None
    assert cfg.binance_api_key.get_secret_value() == "fallback_key"
    assert cfg.binance_api_secret.get_secret_value() == "fallback_secret"


def test_os_env_overrides_dotenv(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LIVE_TRADING=false\nLEVERAGE=1\n", encoding="utf-8")
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("LEVERAGE", "2")
    cfg = AppConfig.from_env()
    assert cfg.live_trading is True
    assert abs(cfg.leverage - 2.0) < 1e-12
