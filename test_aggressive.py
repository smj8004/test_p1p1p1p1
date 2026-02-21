import pandas as pd
import numpy as np

df = pd.read_parquet('data/futures/clean/BTCUSDT/ohlcv_1m.parquet')
if 'open_time' in df.columns:
    df = df.rename(columns={'open_time': 'timestamp'})

# 5m bars for faster trading
m5 = df.resample('5min', on='timestamp').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print('Testing Multiple Aggressive Strategies...')
print()

def test_strategy(m5, strategy_name, leverage, sl_mult, tp_mult, signal_func, max_daily_trades=5):
    initial = 10000
    capital = initial
    fee = 0.0004

    # Indicators
    m5 = m5.copy()
    m5['ema20'] = m5['close'].ewm(span=20).mean()
    m5['ema50'] = m5['close'].ewm(span=50).mean()

    # RSI
    delta = m5['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    m5['rsi'] = 100 - (100 / (1 + gain / loss))

    m5['tr'] = np.maximum(m5['high'] - m5['low'],
                          np.maximum(abs(m5['high'] - m5['close'].shift(1)),
                                     abs(m5['low'] - m5['close'].shift(1))))
    m5['atr'] = m5['tr'].rolling(14).mean()
    m5['vol_ma'] = m5['volume'].rolling(20).mean()
    m5['vol_ratio'] = m5['volume'] / m5['vol_ma']

    # BB
    m5['bb_mid'] = m5['close'].rolling(20).mean()
    m5['bb_std'] = m5['close'].rolling(20).std()
    m5['bb_upper'] = m5['bb_mid'] + 2 * m5['bb_std']
    m5['bb_lower'] = m5['bb_mid'] - 2 * m5['bb_std']
    m5['bb_width'] = (m5['bb_upper'] - m5['bb_lower']) / m5['bb_mid']

    m5['date'] = m5.index.date

    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    daily_trades = {}
    trades = []
    equity = []

    for i in range(60, len(m5)):
        row = m5.iloc[i]
        date = str(row['date'])
        price = row['close']
        atr = row['atr']

        if pd.isna(atr) or atr == 0:
            continue

        if date not in daily_trades:
            daily_trades[date] = 0

        # SL/TP check
        if position == 1:
            if row['low'] <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price * leverage
                capital *= (1 + pnl - fee * 2)
                trades.append({'pnl': pnl * 100, 'type': 'sl'})
                position = 0
            elif row['high'] >= take_profit:
                pnl = (take_profit - entry_price) / entry_price * leverage
                capital *= (1 + pnl - fee * 2)
                trades.append({'pnl': pnl * 100, 'type': 'tp'})
                position = 0
        elif position == -1:
            if row['high'] >= stop_loss:
                pnl = (entry_price - stop_loss) / entry_price * leverage
                capital *= (1 + pnl - fee * 2)
                trades.append({'pnl': pnl * 100, 'type': 'sl'})
                position = 0
            elif row['low'] <= take_profit:
                pnl = (entry_price - take_profit) / entry_price * leverage
                capital *= (1 + pnl - fee * 2)
                trades.append({'pnl': pnl * 100, 'type': 'tp'})
                position = 0

        # Entry
        if position == 0 and daily_trades[date] < max_daily_trades and capital > 0:
            signal = signal_func(m5, i)
            if signal == 1:  # Long
                position = 1
                entry_price = price
                stop_loss = entry_price - atr * sl_mult
                take_profit = entry_price + atr * tp_mult
                daily_trades[date] += 1
            elif signal == -1:  # Short
                position = -1
                entry_price = price
                stop_loss = entry_price + atr * sl_mult
                take_profit = entry_price - atr * tp_mult
                daily_trades[date] += 1

        equity.append(capital)
        if capital <= 0:
            break

    # Close final
    if position != 0 and capital > 0:
        final_price = m5.iloc[-1]['close']
        pnl = (final_price - entry_price) / entry_price * leverage if position == 1 else (entry_price - final_price) / entry_price * leverage
        capital *= (1 + pnl - fee * 2)
        trades.append({'pnl': pnl * 100, 'type': 'final'})

    days = (m5.index[-1] - m5.index[0]).days
    total_return = (capital - initial) / initial * 100 if capital > 0 else -100
    daily_return = total_return / days if days > 0 else 0

    wins = len([t for t in trades if t['pnl'] > 0])

    # Max DD
    if equity:
        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak * 100
        max_dd = dd.min()
    else:
        max_dd = 0

    return {
        'name': strategy_name,
        'leverage': leverage,
        'capital': capital,
        'total_return': total_return,
        'daily_return': daily_return,
        'trades': len(trades),
        'win_rate': wins / len(trades) * 100 if trades else 0,
        'max_dd': max_dd
    }

# Strategy 1: Volume Spike + Momentum
def vol_momentum(m5, i):
    row = m5.iloc[i]
    if pd.isna(row['vol_ratio']) or pd.isna(row['ema20']):
        return 0
    if row['vol_ratio'] > 2.0:
        if row['close'] > row['ema20'] and row['ema20'] > row['ema50']:
            return 1
        elif row['close'] < row['ema20'] and row['ema20'] < row['ema50']:
            return -1
    return 0

# Strategy 2: BB Squeeze Breakout
def bb_squeeze(m5, i):
    row = m5.iloc[i]
    if pd.isna(row['bb_width']):
        return 0
    if row['bb_width'] < 0.02:
        if row['close'] > row['bb_upper']:
            return 1
        elif row['close'] < row['bb_lower']:
            return -1
    return 0

# Strategy 3: RSI Extreme + Volume
def rsi_extreme_vol(m5, i):
    row = m5.iloc[i]
    if pd.isna(row['rsi']) or pd.isna(row['vol_ratio']):
        return 0
    if row['vol_ratio'] > 1.5:
        if row['rsi'] < 25:
            return 1
        elif row['rsi'] > 75:
            return -1
    return 0

# Strategy 4: Scalping (very tight SL/TP)
def scalp_momentum(m5, i):
    row = m5.iloc[i]
    prev = m5.iloc[i-1]
    prev2 = m5.iloc[i-2]

    if pd.isna(row['vol_ratio']):
        return 0

    # 3 consecutive higher closes
    if row['close'] > prev['close'] > prev2['close'] and row['vol_ratio'] > 1.3:
        return 1
    elif row['close'] < prev['close'] < prev2['close'] and row['vol_ratio'] > 1.3:
        return -1
    return 0

# Strategy 5: Mean Reversion
def mean_reversion(m5, i):
    row = m5.iloc[i]
    if pd.isna(row['bb_lower']) or pd.isna(row['rsi']):
        return 0

    # Oversold at lower band
    if row['close'] < row['bb_lower'] and row['rsi'] < 30:
        return 1
    # Overbought at upper band
    elif row['close'] > row['bb_upper'] and row['rsi'] > 70:
        return -1
    return 0

strategies = [
    ('Volume Momentum 5x', 5, 1.5, 2.0, vol_momentum, 5),
    ('Volume Momentum 10x', 10, 1.0, 1.5, vol_momentum, 5),
    ('BB Squeeze 5x', 5, 1.5, 2.0, bb_squeeze, 3),
    ('BB Squeeze 10x', 10, 1.0, 1.5, bb_squeeze, 3),
    ('RSI Extreme 5x', 5, 1.5, 2.5, rsi_extreme_vol, 5),
    ('RSI Extreme 10x', 10, 1.0, 2.0, rsi_extreme_vol, 5),
    ('Scalp 5x', 5, 0.5, 1.0, scalp_momentum, 10),
    ('Scalp 10x', 10, 0.3, 0.6, scalp_momentum, 10),
    ('Mean Reversion 5x', 5, 1.5, 2.0, mean_reversion, 5),
    ('Mean Reversion 10x', 10, 1.0, 1.5, mean_reversion, 5),
]

results = []
for name, lev, sl, tp, func, max_t in strategies:
    r = test_strategy(m5, name, lev, sl, tp, func, max_t)
    results.append(r)
    print(f'{name}: Return {r["total_return"]:+.1f}% | Daily {r["daily_return"]:+.3f}% | WR {r["win_rate"]:.1f}% | Trades {r["trades"]} | MaxDD {r["max_dd"]:.1f}%')

print()
print('=' * 70)
print('BEST STRATEGIES (sorted by daily return)')
print('=' * 70)
for r in sorted(results, key=lambda x: x['daily_return'], reverse=True)[:5]:
    print(f'{r["name"]}: Daily {r["daily_return"]:+.3f}% | Annual ~{r["daily_return"]*365:.0f}%')

print()
print('=' * 70)
print('TARGET: 0.5%/day = 182.5%/year')
print('=' * 70)
best = max(results, key=lambda x: x['daily_return'])
gap = 0.5 - best['daily_return']
print(f'Best achieved: {best["daily_return"]:.3f}%/day')
print(f'Gap to target: {gap:.3f}%/day')
