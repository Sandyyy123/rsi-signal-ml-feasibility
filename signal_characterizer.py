"""
Signal characterization: base rate, forward returns, regime sensitivity.
All operations are point-in-time safe — no future data used.
"""

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_p2t_labels(rsi: pd.Series,
                      trough_threshold: float = 30.0,
                      peak_threshold: float = 70.0) -> pd.Series:
    """
    Label each bar 1 if it is the confirmed START of a trough
    (RSI crosses below trough_threshold after being above peak_threshold).
    Label uses only data available up to that bar — no lookahead.
    """
    labels = pd.Series(0, index=rsi.index, name="label")
    in_trough = False
    prev_was_peak = False

    for i, (idx, val) in enumerate(rsi.items()):
        if np.isnan(val):
            continue
        if val > peak_threshold:
            prev_was_peak = True
            in_trough = False
        elif val < trough_threshold and prev_was_peak and not in_trough:
            labels[idx] = 1
            in_trough = True
        elif val > trough_threshold:
            in_trough = False

    return labels


def characterize_signal(ticker: str, start: str, end: str,
                        rsi_period: int = 14,
                        trough_threshold: float = 30.0,
                        peak_threshold: float = 70.0) -> dict:

    if HAS_YF:
        raw = yf.download(ticker, start=start, end=end, progress=False)
        ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    else:
        # Demo mode: synthetic price series for CI/test environments
        print("  [demo mode] yfinance not available — using synthetic OHLCV")
        dates = pd.date_range(start, end, freq="B")
        np.random.seed(42)
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
        ohlcv = pd.DataFrame({
            "Open": close * np.random.uniform(0.995, 1.0, len(dates)),
            "High": close * np.random.uniform(1.0, 1.015, len(dates)),
            "Low": close * np.random.uniform(0.985, 1.0, len(dates)),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
        }, index=dates)

    rsi = compute_rsi(ohlcv["Close"], period=rsi_period)
    labels = detect_p2t_labels(rsi, trough_threshold, peak_threshold)

    n_signals = labels.sum()
    n_bars = len(labels.dropna())
    base_rate = n_signals / n_bars if n_bars else 0.0

    # Forward returns at 5, 10, 20 days
    fwd = {}
    for h in [5, 10, 20]:
        fwd_ret = ohlcv["Close"].pct_change(h).shift(-h)
        signal_returns = fwd_ret[labels == 1]
        fwd[f"fwd_{h}d_mean"] = float(signal_returns.mean()) if len(signal_returns) else np.nan
        fwd[f"fwd_{h}d_std"] = float(signal_returns.std()) if len(signal_returns) else np.nan

    print(f"  Bars: {n_bars} | Signals: {int(n_signals)} | Base rate: {base_rate:.2%}")
    for h in [5, 10, 20]:
        print(f"  Fwd {h}d mean return: {fwd.get(f'fwd_{h}d_mean', float('nan')):.3%}")

    return {
        "ohlcv": ohlcv,
        "rsi": rsi,
        "labels": labels,
        "base_rate": base_rate,
        "n_signals": int(n_signals),
        "forward_returns": fwd,
        "ticker": ticker,
        "rsi_period": rsi_period,
    }
