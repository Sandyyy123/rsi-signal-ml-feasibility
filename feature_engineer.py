"""
Leakage-safe feature construction for RSI P2T signal.
Every feature uses only data available at prediction time (t-1 or earlier).
"""

import numpy as np
import pandas as pd
from signal_characterizer import compute_rsi


def build_features(ohlcv: pd.DataFrame, labels: pd.Series):
    """
    Returns X (features), y (labels), dates — all aligned, NaN rows dropped.
    All features are lagged by 1 bar minimum to prevent lookahead.
    """
    close = ohlcv["Close"]
    volume = ohlcv["Volume"]
    high = ohlcv["High"]
    low = ohlcv["Low"]

    feats = pd.DataFrame(index=ohlcv.index)

    # RSI at multiple periods (all lagged 1 bar)
    for period in [7, 14, 21]:
        feats[f"rsi_{period}"] = compute_rsi(close, period).shift(1)

    # RSI slope (change over last 3 bars), lagged
    feats["rsi_slope_3"] = compute_rsi(close, 14).diff(3).shift(1)

    # Price momentum (lagged)
    for lookback in [5, 10, 20, 60]:
        feats[f"ret_{lookback}d"] = close.pct_change(lookback).shift(1)

    # Volatility (rolling std of returns, lagged)
    daily_ret = close.pct_change()
    for window in [10, 20]:
        feats[f"vol_{window}d"] = daily_ret.rolling(window).std().shift(1)

    # Volume ratio vs 20-day avg (lagged)
    feats["vol_ratio"] = (volume / volume.rolling(20).mean()).shift(1)

    # ATR-based range (lagged)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feats["atr_14"] = tr.rolling(14).mean().shift(1)

    # Normalized distance from 52-week high/low (lagged)
    feats["dist_52w_high"] = (close / close.rolling(252).max() - 1).shift(1)
    feats["dist_52w_low"] = (close / close.rolling(252).min() - 1).shift(1)

    # Align with labels, drop NaN rows
    df = feats.join(labels.rename("label"), how="inner")
    df = df.dropna()

    X = df.drop(columns=["label"])
    y = df["label"]
    dates = df.index

    print(f"  Features: {X.shape[1]} | Samples: {len(X)} | Positive labels: {int(y.sum())} ({y.mean():.2%})")
    return X, y, dates
