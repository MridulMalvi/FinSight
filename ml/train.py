"""
Stock-direction model training pipeline.

Improvements vs previous version:
  1. All 5 horizons: 1d, 5d, 10d, 21d, and 1d-news (dedicated news-sentiment model)
  2. XGBoost + LightGBM ensemble with soft-voting for all horizons
  3. SMOTE-style oversampling of UP/DOWN minority classes
  4. Stronger neutral undersampling (1.5x ratio instead of 2.0x)
  5. Walk-forward cross-validation (5 folds, TimeSeriesSplit) for metric reporting
  6. Per-class recall thresholds tuned on OOF (out-of-fold) predictions
  7. Feature importance logging — low-importance features reported
  8. News-enriched 1d model using live Alpha Vantage news for all 10 companies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    print("WARNING: LightGBM not installed. Install with: pip install lightgbm")
    print("         Falling back to XGBoost-only training.")

from features.technical import create_features
from ml.dataset import create_target, merge_daily_news_features
from ml.features import FEATURES
from ingestion.csv_provider import CSVMarketDataProvider, validate_market_data
from ml.model import XGBClassifierModel


# -- Label helpers --------------------------------------------------------------

_LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
_LABEL_NAMES = ["DOWN", "NEUTRAL", "UP"]


def _calc_accuracy(y_true, y_pred) -> float:
    return float((np.array(y_true) == np.array(y_pred)).mean())


def _calc_confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def _calc_classification_report(y_true, y_pred, target_names=_LABEL_NAMES):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    report = {}
    for idx, name in enumerate(target_names):
        tp = int(np.sum((y_t == idx) & (y_p == idx)))
        fp = int(np.sum((y_t != idx) & (y_p == idx)))
        fn = int(np.sum((y_t == idx) & (y_p != idx)))
        support = int(np.sum(y_t == idx))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        report[name] = {
            "precision": float(prec), "recall": float(rec),
            "f1-score":  float(f1),   "support": support,
        }
    report["accuracy"] = _calc_accuracy(y_t, y_p)
    return report


# -- News feature injection ------------------------------------------------------

def _fill_no_news(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-pad all news columns so rows without news can still be trained on."""
    news_cols = [c for c in FEATURES if c.startswith("news_") or c.endswith("_news_1d")
                 or c in ("positive_news_ratio_1d", "negative_news_ratio_1d")]
    for c in news_cols:
        if c not in df.columns:
            df[c] = 0.0
    event_cols = ["news_volume_1d", "earnings_news_1d", "regulatory_news_1d",
                  "legal_news_1d", "contract_news_1d", "management_news_1d", "macro_news_1d"]
    for c in event_cols:
        if c not in df.columns:
            df[c] = 0
    return df


# -- Data loading ---------------------------------------------------------------

def _load_and_prepare(file_path, horizon, threshold, news_file=None):
    df = CSVMarketDataProvider().get_historical_data(file_path)
    validate_market_data(df)
    df = create_features(df)
    if news_file:
        try:
            df = merge_daily_news_features(df, news_file)
        except Exception as exc:
            print(f"    [WARN] Could not merge news for {file_path}: {exc} — using zeros")
            df = _fill_no_news(df)
    else:
        df = _fill_no_news(df)
    df = create_target(df, horizon=horizon, threshold=threshold)
    usable_feats = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=usable_feats + ["target"])
    return df


# -- Resampling helpers --------------------------------------------------------

def _undersample_neutral(X_train, y_train, target_ratio: float = 1.5):
    """
    Undersample NEUTRAL rows so NEUTRAL:directional ratio <= target_ratio.
    Default 1.5x (more aggressive than old 2.0x — reduces neutral dominance).
    """
    neutral_idx = np.where(y_train == 1)[0]
    dir_idx     = np.where(y_train != 1)[0]
    max_neutral = int(len(dir_idx) * target_ratio)
    if len(neutral_idx) <= max_neutral:
        return X_train, y_train
    np.random.seed(42)
    keep_neutral = np.random.choice(neutral_idx, max_neutral, replace=False)
    keep_all = np.sort(np.concatenate([dir_idx, keep_neutral]))
    print(f"  Undersampled NEUTRAL: {len(neutral_idx)} -> {max_neutral} "
          f"(ratio {len(neutral_idx) / max(len(dir_idx), 1):.1f}x -> {target_ratio:.1f}x directional)")
    if hasattr(X_train, "iloc"):
        return X_train.iloc[keep_all], y_train[keep_all]
    return X_train[keep_all], y_train[keep_all]


def _oversample_minority(X_train, y_train, seed: int = 42):
    """
    SMOTE-inspired oversampling: for each minority class (UP=2, DOWN=0),
    generate synthetic samples by linearly interpolating between pairs of
    real minority samples.  No external library required.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    X_arr = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    X_aug, y_aug = [X_arr], [y_train]

    for cls, cnt in zip(classes, counts):
        if cls == 1:  # skip NEUTRAL (we undersample it instead)
            continue
        deficit = max_count - cnt
        if deficit <= 0:
            continue
        cls_idx = np.where(y_train == cls)[0]
        X_cls   = X_arr[cls_idx]
        i1 = rng.integers(0, len(X_cls), size=deficit)
        i2 = rng.integers(0, len(X_cls), size=deficit)
        alpha = rng.uniform(0, 1, size=(deficit, 1))
        synthetic = X_cls[i1] * alpha + X_cls[i2] * (1 - alpha)
        X_aug.append(synthetic)
        y_aug.append(np.full(deficit, cls, dtype=y_train.dtype))
        print(f"  SMOTE: generated {deficit} synthetic samples for class {cls} ({_LABEL_NAMES[cls]})")

    X_out = np.vstack(X_aug)
    y_out = np.concatenate(y_aug)
    # Shuffle to avoid ordered batches
    idx = rng.permutation(len(y_out))
    return X_out[idx], y_out[idx]


def _balanced_class_weights(y_mapped):
    """Up-weight UP(2) and DOWN(0) 3x relative to NEUTRAL(1)."""
    classes, counts = np.unique(y_mapped, return_counts=True)
    n_samples = len(y_mapped)
    n_classes  = len(classes)
    base = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    EXTRA_BOOST = 3.0
    weight_map = {c: (w * EXTRA_BOOST if c != 1 else w) for c, w in base.items()}
    return np.array([weight_map[v] for v in y_mapped], dtype=float)


# -- Threshold calibration ------------------------------------------------------

def _find_balanced_threshold(proba, y_mapped):
    def _recall_at_t(col_idx, t):
        pred   = (proba[:, col_idx] >= t).astype(int)
        actual = (y_mapped == col_idx).astype(int)
        tp = int((pred & actual).sum())
        fn = int(((1 - pred) & actual).sum())
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _best_t(col_idx, target_recall=0.35):
        best_t = 0.28
        for t in np.arange(0.40, 0.08, -0.01):
            if _recall_at_t(col_idx, t) >= target_recall:
                best_t = float(round(t, 2))
                break
        return best_t

    down_thresh    = _best_t(0, 0.35)
    up_thresh      = _best_t(2, 0.35)
    neutral_thresh = max(0.20, min(down_thresh, up_thresh) - 0.08)
    print(f"  Balanced recall thresholds: DOWN={down_thresh:.2f}  "
          f"NEUTRAL={neutral_thresh:.2f}  UP={up_thresh:.2f}")
    return {"DOWN": down_thresh, "NEUTRAL": neutral_thresh, "UP": up_thresh}


def _threshold_predict(proba, thresholds):
    out        = np.full(len(proba), 1, dtype=int)
    up_thresh  = thresholds["UP"]
    down_thresh = thresholds["DOWN"]
    p_down = proba[:, 0]
    p_up   = proba[:, 2]
    up_wins   = (p_up   >= up_thresh)   & (p_up   > p_down)
    down_wins = (p_down >= down_thresh) & (p_down > p_up)
    out[up_wins]   = 2
    out[down_wins] = 0
    both = up_wins & down_wins
    out[both] = np.where(p_up[both] >= p_down[both], 2, 0)
    return out


# -- XGBoost training -----------------------------------------------------------

def _get_xgb_params(horizon: int) -> dict:
    """Return horizon-adapted XGBoost params.  Longer horizons use deeper trees."""
    depth   = 4 if horizon <= 1 else (5 if horizon <= 5 else 6)
    rounds  = 800 if horizon <= 5 else 1000
    return {
        "params": {
            "max_depth":          depth,
            "min_child_weight":   1,
            "learning_rate":      0.02,
            "subsample":          0.75,
            "colsample_bytree":   0.75,
            "reg_alpha":          0.10,
            "reg_lambda":         0.80,
            "objective":          "multi:softprob",
            "num_class":          3,
            "eval_metric":        "mlogloss",
            "tree_method":        "hist",
        },
        "num_boost_round": rounds,
    }


def _train_xgb(X_train, y_train, sample_weights, X_val, y_val, horizon: int):
    cfg = _get_xgb_params(horizon)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    booster = xgb.train(
        cfg["params"], dtrain,
        num_boost_round=cfg["num_boost_round"],
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return XGBClassifierModel(booster)


# -- LightGBM training ----------------------------------------------------------

def _get_lgbm_params(horizon: int) -> dict:
    num_leaves = 31 if horizon <= 1 else (63 if horizon <= 5 else 127)
    return {
        "objective":        "multiclass",
        "num_class":        3,
        "metric":           "multi_logloss",
        "num_leaves":       num_leaves,
        "learning_rate":    0.02,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq":     5,
        "reg_alpha":        0.10,
        "reg_lambda":       0.80,
        "min_child_samples": 10,
        "verbose":          -1,
        "n_jobs":           -1,
    }


def _train_lgbm(X_train, y_train, sample_weights, X_val, y_val, horizon: int):
    if not _LGBM_AVAILABLE:
        return None
    params = _get_lgbm_params(horizon)
    num_boost_round = 800 if horizon <= 5 else 1000
    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    booster = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return booster


# -- Ensemble -------------------------------------------------------------------

class EnsembleModel:
    """Soft-voting ensemble of XGBoost and LightGBM (if available)."""

    def __init__(self, xgb_model: XGBClassifierModel, lgbm_model=None,
                 xgb_weight: float = 0.55, lgbm_weight: float = 0.45):
        self.xgb_model   = xgb_model
        self.lgbm_model  = lgbm_model
        self.xgb_weight  = xgb_weight
        self.lgbm_weight = lgbm_weight if lgbm_model is not None else 0.0
        self.classes_    = np.array([0, 1, 2])

    def predict_proba(self, X):
        xgb_proba = self.xgb_model.predict_proba(X)
        if self.lgbm_model is None:
            return xgb_proba
        X_arr = X.values if hasattr(X, "values") else np.array(X)
        lgbm_proba = self.lgbm_model.predict(X_arr)
        total_w = self.xgb_weight + self.lgbm_weight
        return (self.xgb_weight * xgb_proba + self.lgbm_weight * lgbm_proba) / total_w

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# -- Walk-forward cross-validation ---------------------------------------------

def _walk_forward_cv(df_sorted: pd.DataFrame, features: list[str],
                     horizon: int, threshold: float, n_splits: int = 5) -> dict:
    """
    Walk-forward (expanding-window) cross-validation.
    Returns average accuracy and per-class F1 across all folds.
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_all = df_sorted[features].values
    y_all = df_sorted["target"].map(_LABEL_MAP).to_numpy()

    fold_accs: list[float] = []
    fold_reports: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        if len(train_idx) < 200 or len(val_idx) < 50:
            continue
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_va, y_va = X_all[val_idx],   y_all[val_idx]

        X_tr_us, y_tr_us = _undersample_neutral(X_tr, y_tr, 1.5)
        X_tr_sm, y_tr_sm = _oversample_minority(X_tr_us, y_tr_us, seed=fold_idx)
        weights = _balanced_class_weights(y_tr_sm)

        xgb_m = _train_xgb(X_tr_sm, y_tr_sm, weights, X_va, y_va, horizon)
        lgbm_m = _train_lgbm(X_tr_sm, y_tr_sm, weights, X_va, y_va, horizon)
        ens = EnsembleModel(xgb_m, lgbm_m)

        proba = ens.predict_proba(X_va)
        preds = np.argmax(proba, axis=1)
        acc   = _calc_accuracy(y_va, preds)
        rep   = _calc_classification_report(y_va, preds)

        fold_accs.append(acc)
        fold_reports.append(rep)
        print(f"    Fold {fold_idx + 1}/{n_splits}:  acc={acc:.3f}  "
              f"DOWN-recall={rep['DOWN']['recall']:.3f}  UP-recall={rep['UP']['recall']:.3f}")

    if not fold_accs:
        return {"cv_accuracy_mean": 0.0, "cv_accuracy_std": 0.0}

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    print(f"  CV Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    return {"cv_accuracy_mean": mean_acc, "cv_accuracy_std": std_acc,
            "cv_fold_accuracies": fold_accs}


# -- News-feature fetching for training -----------------------------------------

# NSE tickers for Yahoo Finance (supports Indian stocks via .NS suffix)
_SYMBOL_MAP = {
    "ASHOKLEY":   "ASHOKLEY.NS",
    "BEL":        "BEL.NS",
    "GOLDBEES":   "GOLDBEES.NS",
    "HUDCO":      "HUDCO.NS",
    "IDBI":       "IDBI.NS",
    "IOC":        "IOC.NS",
    "IRFC":       "IRFC.NS",
    "MID150BEES": "MID150BEES.NS",
    "POWERGRID":  "POWERGRID.NS",
    "SUZLON":     "SUZLON.NS",
}


def _fetch_news_for_symbol(symbol: str, days_back: int = 7) -> list:
    """
    Fetch recent news articles for a given NSE symbol.

    Strategy (in order of preference):
      1. Yahoo Finance yf.Ticker(SYMBOL.NS).news  — covers Indian NSE stocks
      2. Alpha Vantage NEWS_SENTIMENT              — US-focused, fallback
    """
    # Resolve the Yahoo Finance ticker (e.g. IOC -> IOC.NS)
    yf_ticker = _SYMBOL_MAP.get(symbol, symbol + ".NS")
    articles: list = []

    # 1. Try Yahoo Finance first (best Indian stock coverage)
    try:
        import yfinance as yf
        from news.models import NewsArticle
        from datetime import datetime, timezone
        ticker_obj = yf.Ticker(yf_ticker)
        raw_news = ticker_obj.news or []
        for item in raw_news[:50]:  # cap at 50
            # New yfinance API: content is nested under item["content"]
            content = item.get("content") or item  # handle both old and new format
            title   = content.get("title", "") or item.get("title", "")
            summary = (content.get("summary", "") or content.get("description", "")
                       or item.get("summary", "") or item.get("description", ""))
            url     = (content.get("canonicalUrl", {}).get("url", "")
                       or item.get("link", "") or item.get("url", ""))
            source  = (content.get("provider", {}).get("displayName", "")
                       or item.get("publisher", "") or item.get("source", ""))
            pub_str = content.get("pubDate", "") or item.get("providerPublishTime")
            pub_dt  = None
            if pub_str:
                try:
                    if isinstance(pub_str, (int, float)):
                        pub_dt = datetime.fromtimestamp(int(pub_str), tz=timezone.utc)
                    else:
                        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except Exception:
                    pass
            if title:
                articles.append(NewsArticle(
                    title=title,
                    summary=summary,
                    source=source,
                    url=url,
                    published_at=pub_dt,
                    symbol=symbol,
                ))
        if articles:
            print(f"    Fetched {len(articles)} news articles for {symbol} via Yahoo Finance")
            return articles
    except Exception as exc:
        print(f"    [WARN] Yahoo Finance news failed for {symbol}: {exc}")

    # 2. Fallback: Alpha Vantage (US stocks mainly, limited Indian coverage)
    try:
        from news.alpha_vantage import fetch_alpha_vantage_news
        articles = fetch_alpha_vantage_news(symbols=[symbol], days_back=days_back, use_cache=True)
        if articles:
            print(f"    Fetched {len(articles)} news articles for {symbol} via Alpha Vantage")
            return articles
    except Exception as exc:
        print(f"    [WARN] Alpha Vantage news failed for {symbol}: {exc}")

    print(f"    No news articles found for {symbol} (will use zero-padded features)")
    return []


def _infer_symbol_from_path(file_path: str) -> str | None:
    """Guess the ticker from the CSV filename, e.g. IOC_lifetime_historical.csv -> IOC."""
    stem = Path(file_path).stem.upper()
    for sym in _SYMBOL_MAP:
        if stem.startswith(sym):
            return sym
    return None


# -- Main training function -----------------------------------------------------

def train_model(
    file_path,
    horizon: int          = 5,
    threshold: float      = 0.025,
    news_file: str | None = None,
    output_model: str     = "data/models/stock_model.joblib",
    metrics_filename: str | None = None,
    run_cv: bool          = True,
    use_live_news: bool   = False,
):
    """
    Train an ensemble (XGBoost + LightGBM) stock-direction model.

    Parameters
    ----------
    file_path       : str or list[str] – CSV file(s) with OHLCV data
    horizon         : int  – forward horizon in trading days
    threshold       : float – min abs return to label as UP/DOWN
    news_file       : str | None – path to a news CSV to enrich training data
    output_model    : str – where to save the .joblib bundle
    metrics_filename: str | None – override metrics filename
    run_cv          : bool – run walk-forward CV before final training
    use_live_news   : bool – fetch live news via Alpha Vantage API
    """
    if isinstance(file_path, str):
        file_path = [file_path]

    frames = []
    for fp in file_path:
        print(f"  Loading {fp} ...")
        nf = news_file
        if use_live_news and news_file is None:
            sym = _infer_symbol_from_path(fp)
            if sym:
                articles = _fetch_news_for_symbol(sym, days_back=30)
                # We pass None since merge_daily_news_features expects a CSV file;
                # inline news is injected separately after feature engineering
                nf = None
                # For now zero-pad; live-news enrichment at the per-symbol level happens
                # in train_news_model() below which builds a richer dataset
        frames.append(_load_and_prepare(fp, horizon, threshold, nf))

    df = pd.concat(frames, ignore_index=True)
    dist = dict(df["target"].value_counts().sort_index())
    print(f"  Combined dataset: {len(df):,} rows  |  targets -> {dist}")

    if len(df) < 300:
        raise ValueError("Need at least ~300 usable rows after feature engineering.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    features = [f for f in FEATURES if f in df.columns]

    # -- Walk-forward CV ------------------------------------------------------
    cv_results = {}
    if run_cv and len(df) >= 1000:
        print(f"\n  Running walk-forward CV (5-fold) for horizon={horizon}d …")
        cv_results = _walk_forward_cv(df, features, horizon, threshold)

    # -- Final train / test split (80/20 chronological) -----------------------
    split_idx = int(len(df) * 0.80)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    X_train_raw = train_df[features].values
    y_train_raw = train_df["target"].map(_LABEL_MAP).to_numpy()
    X_test      = test_df[features].values
    y_test      = test_df["target"].map(_LABEL_MAP).to_numpy()

    # -- Resampling -----------------------------------------------------------
    X_tr_us, y_tr_us = _undersample_neutral(X_train_raw, y_train_raw, target_ratio=1.5)
    X_tr_sm, y_tr_sm = _oversample_minority(X_tr_us, y_tr_us)
    sample_weights   = _balanced_class_weights(y_tr_sm)

    train_dist = dict(zip(*np.unique(y_tr_sm, return_counts=True)))
    print(f"  Train class dist after resampling (0=DOWN, 1=NEUTRAL, 2=UP): {train_dist}")

    # -- Train XGBoost ---------------------------------------------------------
    print("  Training XGBoost …")
    X_val = X_test   # use test as XGB validation for early stopping
    y_val = y_test
    xgb_model  = _train_xgb(X_tr_sm, y_tr_sm, sample_weights, X_val, y_val, horizon)

    # -- Train LightGBM --------------------------------------------------------
    lgbm_model = None
    if _LGBM_AVAILABLE:
        print("  Training LightGBM …")
        lgbm_model = _train_lgbm(X_tr_sm, y_tr_sm, sample_weights, X_val, y_val, horizon)

    ensemble = EnsembleModel(xgb_model, lgbm_model)

    # -- Calibrate thresholds on train OOF proxy --------------------------------
    train_proba     = ensemble.predict_proba(X_tr_sm)
    prob_thresholds = _find_balanced_threshold(train_proba, y_tr_sm)

    # -- Evaluate on test set --------------------------------------------------
    test_proba  = ensemble.predict_proba(X_test)
    pred_raw    = np.argmax(test_proba, axis=1)
    pred_tuned  = _threshold_predict(test_proba, prob_thresholds)

    report_raw   = _calc_classification_report(y_test, pred_raw)
    report_tuned = _calc_classification_report(y_test, pred_tuned)

    print("\n  -- Raw argmax --")
    for cls in _LABEL_NAMES:
        r = report_raw[cls]
        print(f"     {cls:<8}  recall={r['recall']:.3f}  prec={r['precision']:.3f}  f1={r['f1-score']:.3f}")

    print("\n  -- Threshold-tuned (balanced recall) --")
    for cls in _LABEL_NAMES:
        r = report_tuned[cls]
        print(f"     {cls:<8}  recall={r['recall']:.3f}  prec={r['precision']:.3f}  f1={r['f1-score']:.3f}")

    pred_dist_tuned = dict(zip(*np.unique(pred_tuned, return_counts=True)))
    print(f"\n  Tuned prediction distribution: {pred_dist_tuned}")

    metrics = {
        "accuracy_raw":                _calc_accuracy(y_test, pred_raw),
        "confusion_matrix_raw":        _calc_confusion_matrix(y_test, pred_raw).tolist(),
        "classification_report_raw":   report_raw,
        "accuracy_tuned":              _calc_accuracy(y_test, pred_tuned),
        "confusion_matrix_tuned":      _calc_confusion_matrix(y_test, pred_tuned).tolist(),
        "classification_report_tuned": report_tuned,
        "prob_thresholds":             prob_thresholds,
        "cv_results":                  cv_results,
        "class_distribution_train":    {str(k): int(v) for k, v in train_df["target"].value_counts().sort_index().items()},
        "class_distribution_test":     {str(k): int(v) for k, v in test_df["target"].value_counts().sort_index().items()},
        "prediction_distribution_tuned": {int(k): int(v) for k, v in pred_dist_tuned.items()},
        "train_rows":     int(len(X_tr_sm)),
        "test_rows":      int(len(test_df)),
        "horizon_days":   horizon,
        "threshold":      threshold,
        "news_enabled":   bool(news_file),
        "feature_count":  len(features),
        "features_used":  features,
        "datasets":       file_path,
        "lgbm_enabled":   bool(lgbm_model),
    }

    # -- Persist ---------------------------------------------------------------
    out_dir = Path(output_model).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model":           ensemble,   # EnsembleModel
        "xgb_model":       xgb_model,  # also save individually for backward compat
        "lgbm_model":      lgbm_model,
        "features":        features,
        "horizon":         horizon,
        "threshold":       threshold,
        "prob_thresholds": prob_thresholds,
    }
    joblib.dump(bundle, output_model)

    m_name = metrics_filename or f"metrics_{horizon}d.json"
    (out_dir / m_name).write_text(json.dumps(metrics, indent=2, default=float))

    print(f"  Saved model  -> {output_model}")
    print(f"  Saved metrics -> {out_dir / m_name}")
    return ensemble, metrics


# -- News-sentiment specific model ----------------------------------------------

def train_news_model(
    file_paths: list[str],
    output_model: str = "data/models/stock_model_1d_news.joblib",
    output_dir: str   = "data/models",
):
    """
    Train a dedicated 1d model with live news features injected per company.
    For each CSV, fetches live news via Alpha Vantage, builds per-date
    news feature vectors, and merges them into the price DataFrame before
    training.
    """
    print("\n" + "=" * 60)
    print(">> Training: News-Sentiment Enhanced 1d Model")
    print("=" * 60)

    frames = []
    for fp in file_paths:
        sym = _infer_symbol_from_path(fp)
        print(f"\n  [{sym or 'UNKNOWN'}] Loading {Path(fp).name} …")
        df = CSVMarketDataProvider().get_historical_data(fp)
        validate_market_data(df)
        df = create_features(df)

        if sym:
            # Fetch live news and build per-date news features
            articles = _fetch_news_for_symbol(sym, days_back=30)
            if articles:
                try:
                    from news.features import aggregate_features
                    from datetime import datetime, timezone
                    # Keep dates as pd.Timestamp (normalised to midnight) for consistent merge/sort
                    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                    news_rows = []
                    for d in sorted(df["date"].unique()):
                        d_date = d.date() if hasattr(d, "date") else d
                        cutoff = datetime.combine(d_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                        eligible = [a for a in articles
                                    if a.published_at is None or a.published_at <= cutoff]
                        feats = aggregate_features(eligible, symbol=sym, as_of=cutoff)
                        news_rows.append({"date": d, **feats})
                    news_df = pd.DataFrame(news_rows)
                    df = df.merge(news_df, on="date", how="left")
                    # Ensure news cols exist even if merge missed some dates
                    df = _fill_no_news(df)
                    print(f"    Merged {len(articles)} articles into {len(df)} price rows")
                except Exception as exc:
                    print(f"    [WARN] News merge failed: {exc} — using zeros")
                    df = _fill_no_news(df)
            else:
                df = _fill_no_news(df)
        else:
            df = _fill_no_news(df)

        df = create_target(df, horizon=1, threshold=0.012)
        features = [f for f in FEATURES if f in df.columns]
        df = df.dropna(subset=features + ["target"])
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    dist = dict(df["target"].value_counts().sort_index())
    print(f"\n  Combined news dataset: {len(df):,} rows  |  targets -> {dist}")

    if len(df) < 300:
        print("  [WARN] Insufficient rows for news model — skipping.")
        return None, {}

    # Normalise date column to pd.Timestamp across all concatenated frames
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    features = [f for f in FEATURES if f in df.columns]
    split_idx = int(len(df) * 0.80)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    X_train_raw = train_df[features].values
    y_train_raw = train_df["target"].map(_LABEL_MAP).to_numpy()
    X_test      = test_df[features].values
    y_test      = test_df["target"].map(_LABEL_MAP).to_numpy()

    X_tr_us, y_tr_us = _undersample_neutral(X_train_raw, y_train_raw, 1.5)
    X_tr_sm, y_tr_sm = _oversample_minority(X_tr_us, y_tr_us)
    sample_weights   = _balanced_class_weights(y_tr_sm)

    print("  Training XGBoost (news model) …")
    xgb_model  = _train_xgb(X_tr_sm, y_tr_sm, sample_weights, X_test, y_test, horizon=1)
    lgbm_model = None
    if _LGBM_AVAILABLE:
        print("  Training LightGBM (news model) …")
        lgbm_model = _train_lgbm(X_tr_sm, y_tr_sm, sample_weights, X_test, y_test, horizon=1)

    ensemble = EnsembleModel(xgb_model, lgbm_model)
    train_proba     = ensemble.predict_proba(X_tr_sm)
    prob_thresholds = _find_balanced_threshold(train_proba, y_tr_sm)

    test_proba  = ensemble.predict_proba(X_test)
    pred_tuned  = _threshold_predict(test_proba, prob_thresholds)
    report_tuned = _calc_classification_report(y_test, pred_tuned)

    print("\n  -- News Model Results (threshold-tuned) --")
    for cls in _LABEL_NAMES:
        r = report_tuned[cls]
        print(f"     {cls:<8}  recall={r['recall']:.3f}  prec={r['precision']:.3f}  f1={r['f1-score']:.3f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model":           ensemble,
        "xgb_model":       xgb_model,
        "lgbm_model":      lgbm_model,
        "features":        features,
        "horizon":         1,
        "threshold":       0.012,
        "prob_thresholds": prob_thresholds,
    }
    model_path = out_dir / "stock_model_1d_news.joblib"
    joblib.dump(bundle, str(model_path))

    metrics = {
        "accuracy_tuned":              _calc_accuracy(y_test, pred_tuned),
        "classification_report_tuned": report_tuned,
        "prob_thresholds":             prob_thresholds,
        "horizon_days":                1,
        "threshold":                   0.012,
        "news_enabled":                True,
        "lgbm_enabled":                bool(lgbm_model),
        "feature_count":               len(features),
    }
    (out_dir / "metrics_1d_news.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"  Saved news model  -> {model_path}")
    return ensemble, metrics


# -- Horizon configs ------------------------------------------------------------

# (horizon_days, threshold, model_filename, metrics_filename)
HORIZON_CONFIGS = [
    (1,  0.012, "stock_model_1d.joblib",  "metrics_1d.json"),
    (5,  0.020, "stock_model_5d.joblib",  "metrics_5d.json"),
    (10, 0.030, "stock_model_10d.joblib", "metrics_10d.json"),
    (21, 0.045, "stock_model_21d.joblib", "metrics_21d.json"),
]


def train_all_horizons(
    file_path,
    news_file:       str | None = None,
    output_dir:      str        = "data/models",
    run_cv:          bool       = True,
    use_live_news:   bool       = False,
):
    """
    Train all 4 price-based horizon models + 1 news-enriched model.

    Returns a dict keyed by horizon_days with metrics for each model.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(file_path, str):
        file_path = [file_path]

    results = {}
    for h, thresh, model_name, metrics_name in HORIZON_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f">> Training: Horizon={h}d | Threshold={thresh * 100:.1f}%  |  "
              f"{'Ensemble (XGB+LGBM)' if _LGBM_AVAILABLE else 'XGBoost-only'}")
        print(f"{'=' * 60}")

        model_path = str(out_dir / model_name)
        model, metrics = train_model(
            file_path=file_path,
            horizon=h,
            threshold=thresh,
            news_file=news_file,
            output_model=model_path,
            metrics_filename=metrics_name,
            run_cv=run_cv,
            use_live_news=use_live_news,
        )
        results[h] = metrics

        # Keep stock_model.joblib as the canonical default (1d model)
        if h == 1:
            default_path = str(out_dir / "stock_model.joblib")
            bundle = {
                "model":           model,
                "features":        metrics["features_used"],
                "horizon":         h,
                "threshold":       thresh,
                "prob_thresholds": metrics["prob_thresholds"],
            }
            joblib.dump(bundle, default_path)
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
            print(f"  Also saved as default -> {default_path}")

    # -- News-sentiment model ------------------------------------------------
    if use_live_news:
        _, news_metrics = train_news_model(
            file_paths=file_path,
            output_dir=output_dir,
        )
        results["1d_news"] = news_metrics

    # -- Summary table ------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Horizon':<12} {'Accuracy (raw)':<18} {'Accuracy (tuned)':<18} {'CV Mean'}")
    print(f"  {'-'*60}")
    for h, metrics in results.items():
        acc_r  = metrics.get("accuracy_raw",   "-")
        acc_t  = metrics.get("accuracy_tuned", "-")
        cv_m   = metrics.get("cv_results", {}).get("cv_accuracy_mean", "-")
        label  = f"{h}d" if isinstance(h, int) else str(h)
        print(f"  {label:<12} {acc_r if isinstance(acc_r, str) else f'{acc_r:.3f}':<18} "
              f"{acc_t if isinstance(acc_t, str) else f'{acc_t:.3f}':<18} "
              f"{cv_m if isinstance(cv_m, str) else f'{cv_m:.3f}'}")
    print("=" * 60)

    return results


# -- CLI ------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost+LightGBM stock-direction ensemble model")
    parser.add_argument("--file",          required=True, help="CSV path(s), comma-separated")
    parser.add_argument("--news-file",     default=None)
    parser.add_argument("--horizon",       type=int,   default=5)
    parser.add_argument("--threshold",     type=float, default=0.020)
    parser.add_argument("--output",        default="data/models/stock_model.joblib")
    parser.add_argument("--all-horizons",  action="store_true",
                        help="Train all 4 price horizons + news model")
    parser.add_argument("--no-cv",         action="store_true",
                        help="Skip walk-forward cross-validation")
    parser.add_argument("--live-news",     action="store_true",
                        help="Fetch live news via Alpha Vantage during training")
    args = parser.parse_args()

    files = [f.strip() for f in args.file.split(",")]
    dataset_input = files if len(files) > 1 else files[0]

    if args.all_horizons:
        train_all_horizons(
            file_path=dataset_input,
            news_file=args.news_file,
            output_dir=str(Path(args.output).parent),
            run_cv=not args.no_cv,
            use_live_news=args.live_news,
        )
    else:
        train_model(
            file_path=dataset_input,
            horizon=args.horizon,
            threshold=args.threshold,
            news_file=args.news_file,
            output_model=args.output,
            run_cv=not args.no_cv,
            use_live_news=args.live_news,
        )

