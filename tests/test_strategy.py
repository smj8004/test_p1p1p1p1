from trader.strategy.base import Bar, StrategyPosition
from trader.strategy.ema_cross import EMACrossStrategy


def test_ema_cross_emits_long_and_short() -> None:
    strategy = EMACrossStrategy(short_window=2, long_window=3)
    closes = [10.0, 10.0, 10.0, 11.0, 12.0, 9.0, 8.0]
    signals = []

    for i, close in enumerate(closes):
        signal = strategy.on_bar(
            Bar(
                timestamp=f"t{i}",
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1_000.0,
            )
        )
        signals.append(signal)

    assert "long" in signals
    assert "short" in signals


def test_ema_cross_emits_exit_on_take_profit() -> None:
    strategy = EMACrossStrategy(short_window=2, long_window=3, take_profit_pct=0.05)
    position = StrategyPosition(side="long", qty=1.0, entry_price=100.0)
    signal = strategy.on_bar(
        Bar(
            timestamp="t0",
            open=105.0,
            high=106.0,
            low=104.0,
            close=106.0,
            volume=1_000.0,
        ),
        position=position,
    )
    assert signal == "exit"
