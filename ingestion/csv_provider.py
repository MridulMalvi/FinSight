import pandas as pd
import numpy as np

REQUIRED = ["date", "open", "high", "low", "close", "volume"]


class CSVMarketDataProvider:
    def get_historical_data(self, file_path: str) -> pd.DataFrame:
        """
        Load OHLCV data from CSV.  Handles three export formats found in the dataset:

        Format A — 8-column header with Date in col 0:
            Date,Price,Adj Close,Close,High,Low,Open,Volume
            (optional blank line)
            2002-07-01,<price>,<adj_close>,<close>,<high>,<low>,<open>,<volume>

        Format B — 7-column, no Date col in header:
            Price,Adj Close,Close,High,Low,Open,Volume
            2009-01-02,<adj_close>,<close>,<high>,<low>,<open>,<volume>

        Format C — Yahoo Finance 3-row multi-header:
            Price,Adj Close,Close,High,Low,Open,Volume
            Ticker,SYMBOL.NS,...
            Date,,,...
            2020-01-01,<adj_close>,<close>,<high>,<low>,<open>,<volume>
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [f.readline() for _ in range(6)]

        stripped = [l.strip() for l in raw_lines]
        non_empty = [l for l in stripped if l]

        df = self._parse(file_path, stripped, non_empty)

        # Standardise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Rename common alternate names
        renames = {
            "adj_close": "adj_close",  # keep as-is
            "price":     "adj_close",  # some formats name adj-close "price"
        }
        df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=REQUIRED).sort_values("date")
        df = df.drop_duplicates("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def _parse(self, file_path, stripped, non_empty):
        h0 = non_empty[0].lower() if non_empty else ""

        # ── Format C: 3-row multi-header (Price / Ticker / Date) ─────────────
        if (h0.startswith("price,") and
                len(non_empty) >= 2 and non_empty[1].lower().startswith("ticker")):
            cols = ["date", "adj_close", "close", "high", "low", "open", "volume"]
            return pd.read_csv(file_path, names=cols, skiprows=3, header=None)

        # ── Format A: "Date,Price,Adj Close,..." header + blank line + 7-col data
        # Data rows have 7 values: date, adj_close, close, high, low, open, volume
        if h0.startswith("date,price,"):
            cols = ["date", "adj_close", "close", "high", "low", "open", "volume"]
            return pd.read_csv(file_path, names=cols, skiprows=2, header=None)

        # ── Format A variant: "Date,...Adj Close,Close,..." without "Price" ────
        if h0.startswith("date,") and "close" in h0:
            return pd.read_csv(file_path)

        # ── Format B: 7-col, first real col is Adj Close ─────────────────────
        if h0.startswith("price,adj close") or h0.startswith("price,adj_close"):
            cols = ["date", "adj_close", "close", "high", "low", "open", "volume"]
            return pd.read_csv(file_path, names=cols, skiprows=1, header=None)

        # ── Generic fallback ──────────────────────────────────────────────────
        return pd.read_csv(file_path)


def validate_market_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty")
    if (df["high"] < df["low"]).any():
        raise ValueError("Invalid OHLC: high < low")
    if (df["close"] <= 0).any():
        raise ValueError("Close must be positive")
    if (df["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
