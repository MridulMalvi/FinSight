"""
Model wrapper classes for the stock prediction ensemble.

XGBClassifierModel
    Lightweight, pickle-safe wrapper around xgb.Booster.
    Compatible with scikit-learn predict/predict_proba API.

EnsembleModel
    Soft-voting ensemble of XGBoost + LightGBM.
    Falls back gracefully when LightGBM is not available.
"""

import numpy as np
import xgboost as xgb


class XGBClassifierModel:
    """
    Lightweight, pickle-safe wrapper around xgb.Booster.
    Does not require scikit-learn C-extensions at runtime.
    """
    def __init__(self, booster):
        self.booster  = booster
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        if isinstance(X, xgb.DMatrix):
            return self.booster.predict(X)
        return self.booster.predict(xgb.DMatrix(X))

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class EnsembleModel:
    """
    Soft-voting ensemble of XGBoost and LightGBM (if available).

    Weights: XGBoost 55%, LightGBM 45% (based on typical F1 parity; can be tuned).
    When LightGBM model is None, falls back to pure XGBoost output.
    """

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
