"""Yahoo Finance Market Data Provider.

Fetches real-time historical OHLCV data for any ticker via the `yfinance` library.
This provider allows predictions without needing a pre-downloaded local CSV file.

Supported tickers:
    US equities:   AAPL, MSFT, GOOGL, TSLA, NVDA, META, AMZN, ...
    Indian NSE:    TCS.NS, RELIANCE.NS, INFY.NS, HDFCBANK.NS, ...
    Indian BSE:    TCS.BO, RELIANCE.BO, ...
    Global:        Any Yahoo Finance ticker (crypto, ETFs, indices)

Usage:
    from ingestion.yfinance_provider import YFinanceMarketDataProvider, validate_yf_ticker

    provider = YFinanceMarketDataProvider()
    df = provider.get_historical_data("AAPL", period="2y")
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Column mapping from yfinance to internal pipeline schema
_YF_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Adj Close": "adj_close",
}

# Known Indian NSE suffix aliases (user may type without .NS)
_KNOWN_NSE = {
    "RELIANCE", "TCS", "INFY", "INFOSYS", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "WIPRO", "AXISBANK",
    "BAJFINANCE", "HINDUNILVR", "ITC", "TITAN", "MARUTI", "ASIANPAINT",
    "SUNPHARMA", "ULTRACEMCO", "TECHM", "NESTLEIND", "POWERGRID",
}

# Map raw CSV stem names to canonical Yahoo Finance tickers
_STEM_TO_YF = {
    "TATAMOTORS": "TMCV.NS",
    "TATAMOTORS.NS": "TMCV.NS",
    "TATAMOTORS.BO": "TMCV.BO",
    "TATA_MOTORS": "TMCV.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFY_NS": "INFY.NS",
    "TCS_NS": "TCS.NS",
    "HDFCBANK_NS": "HDFCBANK.NS",
    "Infosys_historical": "INFY.NS",
    "Reliance_historical": "RELIANCE.NS",
}


def normalize_to_yf(symbol: str) -> str:
    """Convert a raw symbol string to a canonical Yahoo Finance ticker.

    Examples
    --------
    >>> normalize_to_yf("INFY_NS")      # -> 'INFY.NS'
    >>> normalize_to_yf("RELIANCE")     # -> 'RELIANCE.NS'   (known NSE)
    >>> normalize_to_yf("AAPL")         # -> 'AAPL'          (US stock, no change)
    >>> normalize_to_yf("TCS.BO")       # -> 'TCS.BO'        (explicit BSE)
    """
    s = symbol.strip()

    # Direct stem mapping (handles CSV file names like "Infosys_historical")
    if s in _STEM_TO_YF:
        return _STEM_TO_YF[s]

    # Already has an exchange suffix — return as-is
    if "." in s:
        return s

    # Underscore-suffixed Indian symbols (e.g. TCS_NS → TCS.NS)
    if s.upper().endswith("_NS"):
        return s[:-3].upper() + ".NS"
    if s.upper().endswith("_BO") or s.upper().endswith("_BSE"):
        return s.rsplit("_", 1)[0].upper() + ".BO"

    # Strip trailing _HISTORICAL / _HIST labels
    clean = (
        s.upper()
        .replace("_HISTORICAL", "")
        .replace("_HIST", "")
    )

    # Known NSE tickers without suffix
    if clean in _KNOWN_NSE:
        return clean + ".NS"

    # US ticker or unknown — return uppercase as-is
    return clean


def get_yf_ticker_info(symbol: str) -> dict:
    """Return basic info dict from yfinance for a given ticker.

    Returns an empty dict on failure (ticker not found or network error).
    """
    try:
        import yfinance as yf
        yf_sym = normalize_to_yf(symbol)
        t = yf.Ticker(yf_sym)
        info = t.info or {}
        return info
    except Exception as exc:
        logger.warning("yfinance info fetch failed for %s: %s", symbol, exc)
        return {}


def validate_yf_ticker(symbol: str) -> Tuple[bool, str, dict]:
    """Check whether a ticker symbol is valid on Yahoo Finance.

    Returns:
        (is_valid: bool, canonical_symbol: str, metadata: dict)
        metadata has keys: name, sector, currency, exchange, market_cap
    """
    try:
        import yfinance as yf
        yf_sym = normalize_to_yf(symbol)
        t = yf.Ticker(yf_sym)
        info = t.info or {}

        # A valid ticker has at least a regularMarketPrice or previousClose
        price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("currentPrice")
        short_name = info.get("shortName") or info.get("longName") or ""
        if not price and not short_name:
            # Try fast_info as a fallback (yfinance ≥ 0.2)
            try:
                fi = t.fast_info
                price = getattr(fi, "last_price", None)
            except Exception:
                pass

        is_valid = bool(price or short_name)
        meta = {
            "name": short_name,
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "currency": info.get("currency", ""),
            "exchange": info.get("exchange", ""),
            "market_cap": info.get("marketCap"),
        }
        return is_valid, yf_sym, meta
    except Exception as exc:
        logger.warning("validate_yf_ticker failed for %s: %s", symbol, exc)
        return False, normalize_to_yf(symbol), {}


class YFinanceMarketDataProvider:
    """Market data provider backed by Yahoo Finance via the yfinance library.

    Fetches historical OHLCV data and normalizes it to the same schema
    used by CSVMarketDataProvider so the ML pipeline works transparently.
    """

    REQUIRED = ["date", "open", "high", "low", "close", "volume"]

    def get_historical_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol:   Stock ticker (e.g. 'AAPL', 'RELIANCE.NS', 'TCS_NS').
            period:   Data range string accepted by yfinance (e.g. '1y', '2y', '5y').
                      Using '2y' by default to have enough data for the ML model.
            interval: Bar interval ('1d' for daily data).

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Sorted ascending by date, duplicates dropped.

        Raises:
            ValueError: If the ticker is invalid or no data is returned.
            ImportError: If the yfinance package is not installed.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for live data fetching. "
                "Install it with: pip install yfinance>=0.2.40"
            ) from exc

        yf_sym = normalize_to_yf(symbol)
        logger.info("Fetching Yahoo Finance data for %s (normalized: %s), period=%s", symbol, yf_sym, period)

        try:
            ticker = yf.Ticker(yf_sym)
            df_raw = ticker.history(period=period, interval=interval, auto_adjust=True)
        except Exception as exc:
            raise ValueError(f"Failed to fetch Yahoo Finance data for '{yf_sym}': {exc}") from exc

        if df_raw is None or df_raw.empty:
            raise ValueError(
                f"No historical data returned from Yahoo Finance for ticker '{yf_sym}'. "
                f"The ticker may be invalid or delisted."
            )

        # Reset index: yfinance returns DatetimeIndex
        df = df_raw.reset_index()

        # Rename columns to lowercase pipeline schema
        df.columns = [
            _YF_COLUMN_MAP.get(c, c.lower().replace(" ", "_"))
            for c in df.columns
        ]

        # Normalize the date column (may be 'date' or 'datetime' depending on interval)
        if "datetime" in df.columns and "date" not in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        # Convert date to timezone-naive datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)

        # Ensure numeric types for OHLCV
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validate required columns
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Yahoo Finance data missing required columns: {missing}. Got: {list(df.columns)}")

        # Clean
        df = df.dropna(subset=self.REQUIRED)
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

        if df.empty:
            raise ValueError(f"All data rows dropped after cleaning for ticker '{yf_sym}'.")

        logger.info("Yahoo Finance: %d rows fetched for %s (%s to %s)",
                    len(df), yf_sym,
                    df["date"].iloc[0].strftime("%Y-%m-%d"),
                    df["date"].iloc[-1].strftime("%Y-%m-%d"))
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return the most recent close price for a ticker."""
        try:
            df = self.get_historical_data(symbol, period="5d")
            return float(df["close"].iloc[-1])
        except Exception as exc:
            logger.warning("Could not fetch latest price for %s: %s", symbol, exc)
            return None
