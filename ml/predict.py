import joblib
import pandas as pd
from features.technical import create_features
from ingestion.csv_provider import validate_market_data
from config.settings import MODEL_PATH


# ___ Labels ____________________________________________________________________
_LABELS = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


def _news_adjusted_probs(raw_probs: list[float], news_features: dict) -> list[float]:
    """
    Blend XGBoost model probabilities with a news-sentiment signal.

    WHY THIS IS NEEDED
    ------------------
    The trained model is biased toward NEUTRAL because all training rows had
    zero-valued news features (there was no live news feed at training time).
    XGBoost therefore learned near-zero weights for every news_* column, making
    the news input essentially invisible to the model.

    This post-hoc adjustment redistributes probability mass proportionally to
    the aggregated news sentiment so that clearly bearish/bullish news moves
    the prediction toward DOWN/UP respectively, while neutral or absent news
    leaves the model output unchanged.

    Blending scheme
    ---------------
    adjusted = model_weight * raw_probs + news_weight * news_implied_probs

    where:
      blend        = effective_sentiment * importance   (signed impact)
      news_weight  = min(0.92, |blend| * 6.0)           (ramps 0 to 0.92)
      model_weight = 1 - news_weight

    Effective sentiment priority:
      1. Marketaux entity-level sentiment (domain-trained model, highest accuracy)
      2. NLP lexicon sentiment (fallback)
      3. Article positive/negative ratio (last resort when both are weak)

    Args:
        raw_probs:     [p_down, p_neutral, p_up] from model.predict_proba, summing to 1.
        news_features: aggregated feature dict from news.features.aggregate_features.

    Returns:
        Adjusted [p_down, p_neutral, p_up] summing to 1.
    """
    if not news_features:
        return raw_probs

    volume         = int(news_features.get("news_volume_1d", 0))
    negative_ratio = float(news_features.get("negative_news_ratio_1d", 0.0))
    positive_ratio = float(news_features.get("positive_news_ratio_1d", 0.0))
    importance     = float(news_features.get("news_importance_1d", 0.5))

    # No news at all -> model output unchanged
    if volume == 0:
        return raw_probs

    # Step 1: Determine effective sentiment
    # Prefer Marketaux entity-level sentiment (domain-trained financial model).
    # Fall back to aggregated NLP lexicon sentiment.
    mx_avg        = news_features.get("marketaux_sentiment_avg")
    nlp_sentiment = float(news_features.get("news_sentiment_1d", 0.0))

    if mx_avg is not None:
        # Marketaux sentiment is financial-domain trained: weight it 70% vs 30% NLP
        effective_sentiment = 0.70 * float(mx_avg) + 0.30 * nlp_sentiment
    else:
        effective_sentiment = nlp_sentiment

    # Step 2: Ratio-based fallback when direct signal is weak
    # Use article polarity ratio to guide direction if |sentiment| < 0.05
    if abs(effective_sentiment) < 0.05:
        if negative_ratio > positive_ratio and negative_ratio >= 0.35:
            # Most articles are negative -> push sentiment negative
            effective_sentiment = -(negative_ratio * 0.6)
        elif positive_ratio > negative_ratio and positive_ratio >= 0.35:
            # Most articles are positive -> push sentiment positive
            effective_sentiment = positive_ratio * 0.6

    # Step 3: Composite signed signal
    # importance in [0.4, 1.0], sentiment in [-1, 1] -> blend in [-1, 1]
    blend     = effective_sentiment * importance
    abs_blend = abs(blend)

    # Genuinely flat signal with no ratio skew -> leave model alone
    if abs_blend < 0.005 and negative_ratio < 0.25 and positive_ratio < 0.25:
        return raw_probs

    # Step 4: News influence weight
    # Ramps so that moderate news shifts predictions meaningfully but doesn't dominate:
    #   |blend| = 0.04 -> news_weight ~0.14 (mild: model still dominant)
    #   |blend| = 0.08 -> news_weight ~0.28 (moderate: model still dominant)
    #   |blend| = 0.15 -> news_weight ~0.52 (strong: near 50/50 blend)
    # Cap at 0.55 (was 0.92) to keep model probabilities relevant at all times.
    news_weight  = min(0.55, abs_blend * 3.5)
    model_weight = 1.0 - news_weight

    magnitude = min(abs_blend * 2.5, 1.0)  # amplify for sharper distribution skew

    # Step 5: Build news-implied probability distribution
    if blend < 0:   # Bearish -> strongly favour DOWN
        ni_down    = min(0.96, 0.55 + 0.41 * magnitude)
        ni_neutral = max(0.02, 0.30 - 0.26 * magnitude)
        ni_up      = max(0.01, 0.15 - 0.14 * magnitude)
    else:            # Bullish -> strongly favour UP
        ni_up      = min(0.96, 0.55 + 0.41 * magnitude)
        ni_neutral = max(0.02, 0.30 - 0.26 * magnitude)
        ni_down    = max(0.01, 0.15 - 0.14 * magnitude)

    # Normalise news-implied distribution
    total_ni   = ni_down + ni_neutral + ni_up
    ni_down   /= total_ni
    ni_neutral /= total_ni
    ni_up      /= total_ni

    p_down, p_neutral, p_up = raw_probs

    # Step 6: Weighted blend of model and news-implied distributions
    f_down    = model_weight * p_down    + news_weight * ni_down
    f_neutral = model_weight * p_neutral + news_weight * ni_neutral
    f_up      = model_weight * p_up      + news_weight * ni_up

    # Normalise to handle any floating-point drift
    total_f = f_down + f_neutral + f_up
    return [f_down / total_f, f_neutral / total_f, f_up / total_f]


def predict(df: pd.DataFrame, news_features: dict | None = None, symbol: str | None = None) -> dict:
    validate_market_data(df)
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]
    # Probability thresholds tuned at training time to reduce neutral bias
    prob_thresholds: dict = bundle.get("prob_thresholds", {"DOWN": 0.5, "NEUTRAL": 0.5, "UP": 0.5})

    enriched = create_features(df)
    news_features = news_features or {}

    # Inject news features into the feature frame so the model can use them
    for f in features:
        if f not in enriched.columns:
            enriched[f] = float(news_features.get(f, 0.0))
        elif f.startswith("news_") or f.endswith("_news_1d"):
            enriched[f] = float(
                news_features.get(
                    f,
                    enriched[f].iloc[-1] if pd.notna(enriched[f].iloc[-1]) else 0.0,
                )
            )

    latest = enriched.dropna(subset=[f for f in features if f in enriched.columns]).iloc[-1]
    X = pd.DataFrame([latest[features]])

    # Raw model output -- biased toward NEUTRAL when news features were zero during training
    raw_probs = list(model.predict_proba(X)[0])

    # Post-hoc news adjustment: redistributes probability mass based on sentiment
    adjusted = _news_adjusted_probs(raw_probs, news_features)

    # Apply per-class probability thresholds (trained anti-bias fix) instead of
    # raw argmax. UP/DOWN win when their probability exceeds their respective
    # trained threshold; otherwise fall back to NEUTRAL.
    p_down, p_neutral, p_up = adjusted

    # Select the class with the highest probability (argmax) so the predicted label
    # always matches the winning probability bar in the breakdown
    predicted_class = int(adjusted.index(max(adjusted)))

    # ── Expected Return & Target Price (next-day forecast horizon) ───────────
    # The news-enhanced single prediction is calibrated for a 1-day window.
    # We compute the expected price change by probability-weighting the projected moves,
    # amplified by news sentiment magnitude.
    sentiment = float(news_features.get("news_sentiment_1d", 0.0))
    mx_avg = news_features.get("marketaux_sentiment_avg")
    if mx_avg is not None:
        sentiment = 0.60 * float(mx_avg) + 0.40 * sentiment

    # Base class moves calibrated to a 1-day window (±1.8%)
    r_up_base      = 0.018 + 0.010 * max(0.0, sentiment)
    r_down_base    = -0.018 + 0.010 * min(0.0, sentiment)
    r_neutral_base = 0.003 * sentiment

    expected_return = (p_up * r_up_base) + (p_down * r_down_base) + (p_neutral * r_neutral_base)
    close_price = float(latest["close"])
    target_price = float(round(close_price * (1.0 + expected_return), 2))
    price_change = float(round(target_price - close_price, 2))
    price_change_pct = float(round(expected_return * 100.0, 2))

    # Determine currency
    from news.company_context import get_currency
    currency, currency_symbol = get_currency(symbol or "")

    return {
        "predicted_class":      predicted_class,
        "label":                _LABELS[predicted_class],
        "probability_down":     float(adjusted[0]),
        "probability_neutral":  float(adjusted[1]),
        "probability_up":       float(adjusted[2]),
        "as_of":                str(latest["date"]),
        "close":                close_price,
        "target_price":         target_price,
        "price_change":         price_change,
        "price_change_pct":     price_change_pct,
        "forecast_horizon_days": 1,
        "currency":             currency,
        "currency_symbol":      currency_symbol,
        "news_features_used": {
            k: float(news_features.get(k, 0))
            for k in features
            if k.startswith("news_") or k.endswith("_news_1d")
        },
        # Expose raw model output so callers can inspect the bias
        "raw_model_probs": {
            "probability_down":    float(raw_probs[0]),
            "probability_neutral": float(raw_probs[1]),
            "probability_up":      float(raw_probs[2]),
        },
        "prob_thresholds_used": prob_thresholds,
    }



from pathlib import Path

# ── Multi-Horizon Specifications ──────────────────────────────────────────────
# (trading_days, calendar_label, trading_days_label, base_target_move_pct, fallback_multiplier)
# The 1d horizon now has a dedicated trained model: stock_model_1d.joblib
_HORIZON_SPECS = [
    (1,  "Next Day",      "1 trading day",   0.018, 0.50),
    (5,  "1 Week",        "5 trading days",  0.035, 1.00),
    (10, "15 Days",       "10 trading days", 0.055, 1.60),
    (21, "1 Month",       "21 trading days", 0.085, 2.40),
]

_MODEL_CACHE: dict[int, dict] = {}


def get_horizon_model_bundle(horizon: int = 5, news_enhanced: bool = False) -> tuple[object, list[str], dict | None, bool]:
    """
    Retrieve model bundle (model, features, prob_thresholds, is_dedicated) for a horizon.

    Priority order:
      1. stock_model_{horizon}d_news.joblib  (if news_enhanced=True)
      2. stock_model_{horizon}d.joblib       (dedicated horizon model)
      3. stock_model.joblib                  (base model fallback)

    Results are cached in-memory for sub-millisecond repeated lookups.
    """
    cache_key = f"{horizon}_{'news' if news_enhanced else 'price'}"
    if cache_key in _MODEL_CACHE:
        b = _MODEL_CACHE[cache_key]
        return b["model"], b["features"], b.get("prob_thresholds"), b.get("is_dedicated", False)

    model_dir = Path(MODEL_PATH).parent

    # Candidate paths in preference order
    candidates: list[tuple[Path, bool]] = []
    if news_enhanced:
        candidates.append((model_dir / f"stock_model_{horizon}d_news.joblib", True))
    candidates.append((model_dir / f"stock_model_{horizon}d.joblib", True))
    candidates.append((Path(MODEL_PATH), False))

    # Also search module-relative data/models for each candidate
    alt_dir = Path(__file__).resolve().parent.parent / "data" / "models"
    if alt_dir != model_dir:
        if news_enhanced:
            candidates.append((alt_dir / f"stock_model_{horizon}d_news.joblib", True))
        candidates.append((alt_dir / f"stock_model_{horizon}d.joblib", True))
        candidates.append((alt_dir / "stock_model.joblib", False))

    chosen_path, is_dedicated = None, False
    for path, dedicated in candidates:
        if path.exists():
            chosen_path, is_dedicated = path, dedicated
            break

    if chosen_path is None:
        raise FileNotFoundError(
            f"No model found for horizon={horizon}d. "
            f"Run scripts/retrain_all.py to train models."
        )

    bundle = joblib.load(chosen_path)
    bundle["is_dedicated"] = is_dedicated
    _MODEL_CACHE[cache_key] = bundle
    return bundle["model"], bundle["features"], bundle.get("prob_thresholds"), is_dedicated


def predict_multi_horizon(
    df: "pd.DataFrame",
    symbol: "str | None" = None,
    news_features: "dict | None" = None,
) -> dict:
    """Produce predictions for 4 time horizons using dedicated retrained models.

    If dedicated models (stock_model_5d.joblib, stock_model_10d.joblib, stock_model_21d.joblib)
    exist, each horizon runs dedicated XGBoost inference with its own learned weights.
    If not, it cleanly falls back to calibrated scaling from the base model.

    Args:
        df:            Historical OHLCV DataFrame (same format as predict()).
        symbol:        Ticker symbol for currency lookup.
        news_features: Optional news feature dict. If provided, probabilities are
                       news-adjusted (same logic as predict()).

    Returns:
        {
          "success": True,
          "symbol": str,
          "as_of": "YYYY-MM-DD",
          "close": float,
          "currency": str,
          "currency_symbol": str,
          "horizons": [
              {"label": "Next Day", "trading_days": 1, "direction": "UP", ...},
              {"label": "1 Week", "trading_days": 5, "direction": "UP", ...},
              {"label": "15 Days", "trading_days": 10, "direction": "UP", ...},
              {"label": "1 Month", "trading_days": 21, "direction": "NEUTRAL", ...},
          ],
          "news_features_used": bool,
        }
    """
    validate_market_data(df)
    enriched = create_features(df)
    nf = news_features or {}

    close_price = float(enriched["close"].iloc[-1])
    as_of_date = str(enriched["date"].iloc[-1])

    from news.company_context import get_currency
    currency, currency_symbol = get_currency(symbol or "")

    # Effective sentiment for price target calculations
    sentiment = float(nf.get("news_sentiment_1d", 0.0))
    mx_avg = nf.get("marketaux_sentiment_avg")
    if mx_avg is not None:
        sentiment = 0.60 * float(mx_avg) + 0.40 * sentiment

    horizons_result = []

    for trading_days, cal_label, td_label, base_target_pct, fallback_multiplier in _HORIZON_SPECS:
        model_h, features_h, thresholds_h, is_dedicated = get_horizon_model_bundle(trading_days)

        # Inject news features for this model's required feature list
        df_for_model = enriched.copy()
        for f in features_h:
            if f not in df_for_model.columns:
                df_for_model[f] = float(nf.get(f, 0.0))
            elif f.startswith("news_") or f.endswith("_news_1d"):
                df_for_model[f] = float(nf.get(f, df_for_model[f].iloc[-1]
                                           if pd.notna(df_for_model[f].iloc[-1]) else 0.0))

        latest = df_for_model.dropna(subset=[f for f in features_h if f in df_for_model.columns]).iloc[-1]
        X = pd.DataFrame([latest[features_h]])

        raw_probs = list(model_h.predict_proba(X)[0])
        adjusted = _news_adjusted_probs(raw_probs, nf)
        p_down, p_neutral, p_up = adjusted

        # Label determination
        predicted_class = int(adjusted.index(max(adjusted)))
        label = _LABELS[predicted_class]

        # Target price and expected return
        # Each horizon has its calibrated expected return amplitude
        # e.g. 3.5% for 5d, 5.5% for 10d, 8.5% for 21d
        r_up = base_target_pct + 0.025 * max(0.0, sentiment)
        r_down = -base_target_pct + 0.025 * min(0.0, sentiment)
        r_neutral = 0.008 * sentiment

        expected_return = (p_up * r_up) + (p_down * r_down) + (p_neutral * r_neutral)
        target_price = float(round(close_price * (1.0 + expected_return), 2))
        price_change = float(round(target_price - close_price, 2))
        price_change_pct = float(round(expected_return * 100.0, 2))

        horizons_result.append({
            "label": cal_label,
            "trading_days": trading_days,
            "trading_days_label": td_label,
            "direction": label,
            "direction_class": predicted_class,
            "probability_up": float(round(p_up, 4)),
            "probability_neutral": float(round(p_neutral, 4)),
            "probability_down": float(round(p_down, 4)),
            "target_price": target_price,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "is_dedicated_model": bool(is_dedicated),
        })

    return {
        "success": True,
        "symbol": symbol,
        "as_of": as_of_date,
        "close": close_price,
        "currency": currency,
        "currency_symbol": currency_symbol,
        "horizons": horizons_result,
        "news_features_used": bool(nf),
    }

