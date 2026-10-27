# ── Price & Return Features ───────────────────────────────────────────────────
FEATURES = [
    # Core returns
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_30d", "return_60d",

    # Momentum & Oscillators
    "rsi", "macd", "macd_signal", "macd_hist",
    "stoch", "cci", "williams_r",

    # Volatility & Range
    "bb_width", "bb_pct", "atr_pct", "volatility_20", "daily_range",

    # Trend Strength
    "adx",

    # Moving Average relative position
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",

    # Volume
    "volume_ratio", "obv_norm",

    # Ichimoku
    "ichimoku_cloud_direction", "price_vs_cloud",

    # 52-week range position
    "pos_in_52w_range", "pct_from_52w_high",

    # Momentum & Acceleration
    "momentum_10d", "price_accel",

    # ── News Intelligence Features ────────────────────────────────────────────
    # Populated by the daily news pipeline; zero-padded when no news.
    "news_sentiment_1d", "news_impact_1d", "news_importance_1d",
    "news_volume_1d", "positive_news_ratio_1d", "negative_news_ratio_1d",
    "earnings_news_1d", "regulatory_news_1d", "legal_news_1d",
    "contract_news_1d", "management_news_1d", "macro_news_1d",
]
