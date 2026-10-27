import pandas as pd

from news.ingest import load_news_csv
from news.features import aggregate_features


def create_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.03):
    out = df.copy()
    future = out["close"].shift(-horizon)
    out["future_return"] = future / out["close"] - 1.0
    out["target"] = 0
    out.loc[out["future_return"] > threshold, "target"] = 1
    out.loc[out["future_return"] < -threshold, "target"] = -1
    return out


def merge_daily_news_features(price_df: pd.DataFrame, news_csv: str, symbol: str | None = None) -> pd.DataFrame:
    """Point-in-time safe merge: each trading day sees only news published by that date."""
    articles = load_news_csv(news_csv)
    out = price_df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    rows = []
    for d in sorted(out["date"].unique()):
        cutoff = pd.Timestamp(d).to_pydatetime()
        if cutoff.tzinfo is None:
            from datetime import timezone
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        eligible = [a for a in articles if a.published_at is None or a.published_at <= cutoff]
        rows.append({"date": d, **aggregate_features(eligible, symbol=symbol, as_of=cutoff)})
    news_df = pd.DataFrame(rows)
    return out.merge(news_df, on="date", how="left")
