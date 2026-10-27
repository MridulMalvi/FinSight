"""
Technical feature engineering for stock price prediction.

Indicators computed:
  - Price returns: 1d, 3d, 5d, 10d, 20d, 30d, 60d
  - SMAs: 20, 50, 200
  - EMA: 20
  - RSI (14)
  - MACD + signal + histogram
  - Bollinger Bands width
  - ATR % (14)
  - ADX (14)
  - Stochastic Oscillator (14)
  - Volume ratio vs 20-day SMA
  - Volatility (20-day rolling std of 1d returns)
  - Price vs SMA20/50/200
  - Daily range
  - CCI (20)
  - Williams %R (14)
  - OBV (On Balance Volume) normalised
  - Ichimoku cloud signals (tenkan, kijun, leading span A & B, cloud direction)
  - 52-week high/low position
  - Momentum (10d)
  - Price acceleration (return_1d - return_1d_lag5)
"""

import pandas as pd
import numpy as np
import ta


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Price Returns ────────────────────────────────────────────────────────────
    df["return_1d"]  = df["close"].pct_change(1)
    df["return_3d"]  = df["close"].pct_change(3)
    df["return_5d"]  = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)
    df["return_30d"] = df["close"].pct_change(30)
    df["return_60d"] = df["close"].pct_change(60)

    # ── Moving Averages ──────────────────────────────────────────────────────────
    df["sma_20"]  = df["close"].rolling(20).mean()
    df["sma_50"]  = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["ema_20"]  = df["close"].ewm(span=20, adjust=False).mean()

    # ── RSI ─────────────────────────────────────────────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # ── MACD ────────────────────────────────────────────────────────────────────
    macd = ta.trend.MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # ── Bollinger Bands ──────────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"].replace(0, np.nan)
    # Bollinger %B: position within the band (0 = lower, 1 = upper)
    df["bb_pct"]   = bb.bollinger_pband()

    # ── ATR ─────────────────────────────────────────────────────────────────────
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"]     = atr.average_true_range()
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

    # ── ADX ─────────────────────────────────────────────────────────────────────
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()

    # ── Stochastic Oscillator ────────────────────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    df["stoch"] = stoch.stoch()

    # ── Volume Indicators ────────────────────────────────────────────────────────
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"]  = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

    # OBV: accumulate based on price direction
    obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"])
    df["obv_raw"] = obv.on_balance_volume()
    # Normalise OBV to % change vs its 20-day average (removes scale differences between stocks)
    obv_sma = df["obv_raw"].rolling(20).mean()
    df["obv_norm"] = (df["obv_raw"] - obv_sma) / (obv_sma.abs().replace(0, np.nan))

    # ── Volatility ───────────────────────────────────────────────────────────────
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # ── Price vs SMAs ─────────────────────────────────────────────────────────────
    df["price_vs_sma20"]  = df["close"] / df["sma_20"].replace(0, np.nan)  - 1
    df["price_vs_sma50"]  = df["close"] / df["sma_50"].replace(0, np.nan)  - 1
    df["price_vs_sma200"] = df["close"] / df["sma_200"].replace(0, np.nan) - 1

    # ── Daily Range ──────────────────────────────────────────────────────────────
    df["daily_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # ── CCI (Commodity Channel Index) ─────────────────────────────────────────────
    cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20)
    df["cci"] = cci.cci()

    # ── Williams %R ──────────────────────────────────────────────────────────────
    wr = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14)
    df["williams_r"] = wr.williams_r()

    # ── Ichimoku Cloud ────────────────────────────────────────────────────────────
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_high = df["high"].rolling(9).max()
    tenkan_low  = df["low"].rolling(9).min()
    df["ichimoku_tenkan"] = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_high = df["high"].rolling(26).max()
    kijun_low  = df["low"].rolling(26).min()
    df["ichimoku_kijun"] = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods forward
    span_a = (df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2
    df["ichimoku_span_a"] = span_a.shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods
    span_b_high = df["high"].rolling(52).max()
    span_b_low  = df["low"].rolling(52).min()
    df["ichimoku_span_b"] = ((span_b_high + span_b_low) / 2).shift(26)

    # Cloud direction: +1 if span_a > span_b (bullish cloud), -1 otherwise
    df["ichimoku_cloud_direction"] = np.where(
        df["ichimoku_span_a"] > df["ichimoku_span_b"], 1.0, -1.0
    )
    # Price relative to cloud (vs midpoint of span_a and span_b)
    cloud_mid = (df["ichimoku_span_a"] + df["ichimoku_span_b"]) / 2
    df["price_vs_cloud"] = (df["close"] - cloud_mid) / cloud_mid.replace(0, np.nan)

    # ── 52-Week High/Low Position ─────────────────────────────────────────────────
    rolling_52w_high = df["high"].rolling(252, min_periods=50).max()
    rolling_52w_low  = df["low"].rolling(252, min_periods=50).min()
    span_52w = (rolling_52w_high - rolling_52w_low).replace(0, np.nan)
    # 0.0 = at 52-week low, 1.0 = at 52-week high
    df["pos_in_52w_range"] = (df["close"] - rolling_52w_low) / span_52w
    # Distance from 52-week high as %
    df["pct_from_52w_high"] = (df["close"] / rolling_52w_high.replace(0, np.nan)) - 1.0

    # ── Momentum ──────────────────────────────────────────────────────────────────
    # Rate-of-change momentum: close vs 10 days ago
    df["momentum_10d"] = df["close"].pct_change(10)
    # Price acceleration: recent 1d return vs 5d ago 1d return
    df["price_accel"] = df["return_1d"] - df["return_1d"].shift(5)

    return df
