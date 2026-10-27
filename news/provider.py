"""News Provider interface for FinSight AI Engine.

Combines Yahoo Finance Real-Time News and Alpha Vantage News & Sentiment API.
Provides resilient real-time news retrieval for global and Indian stocks
with automatic fallback and multi-tiered caching.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple
from dotenv import load_dotenv

from news.models import NewsArticle
from news.alpha_vantage import fetch_alpha_vantage_news, invalidate_av_cache
from news.yahoo import fetch_yahoo_news, invalidate_yf_cache

logger = logging.getLogger(__name__)


def get_active_provider_info() -> dict:
    """Return status of configured news providers."""
    load_dotenv(override=True)
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "") or os.getenv("ALPHAVANTAGE_API_KEY", "")

    return {
        "active_provider": "hybrid",
        "yahoo_finance_available": True,
        "alphavantage_configured": bool(av_key),
    }


def is_non_us_or_yf_symbol(symbol: str) -> bool:
    """Check if symbol has exchange suffixes (e.g., .NS, .BO, .BSE, .L) or is known Indian stock."""
    s = symbol.strip().upper()
    if any(s.endswith(sfx) for sfx in [".NS", ".BO", ".BSE", ".L", ".TO", ".AX", ".HK", ".SS", ".SZ"]):
        return True
    if any(tag in s for tag in ["_NS", "_BO", "TCS", "RELIANCE", "INFY", "HDFCBANK", "TATAMOTORS", "ICICIBANK"]):
        return True
    return False


def fetch_live_news(
    symbols: List[str],
    days_back: int = 2,
    use_cache: bool = True,
) -> Tuple[List[NewsArticle], str]:
    """Fetch live news with multi-provider resilience.

    Strategy:
    1. If symbol is an Indian/international ticker with exchange suffix or non-US,
       fetch directly from Yahoo Finance news (guaranteed real-time company coverage).
    2. If symbol is US ticker, attempt Alpha Vantage news first if configured.
       If Alpha Vantage returns articles, use them.
    3. If Alpha Vantage returns 0 articles, hits rate limits, or fails,
       immediately fall back to Yahoo Finance news.
    4. If network is offline and both yield 0, check curated offline fallback.

    Returns:
        (articles, provider_name)
    """
    load_dotenv(override=True)
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "") or os.getenv("ALPHAVANTAGE_API_KEY", "")

    clean_symbols = [s.strip() for s in symbols if s.strip()]
    if not clean_symbols:
        return [], "none"

    first_sym = clean_symbols[0]

    # Case 1: Indian or non-US tickers (Alpha Vantage returns 0 or rejects dots)
    if is_non_us_or_yf_symbol(first_sym):
        logger.info("Fetching news for %s via Yahoo Finance News...", clean_symbols)
        articles = fetch_yahoo_news(clean_symbols, days_back=days_back, use_cache=use_cache)
        if articles:
            return articles, "yahoo_finance"

        # If Yahoo returns 0, try Alpha Vantage as secondary
        if av_key:
            try:
                av_articles = fetch_alpha_vantage_news(clean_symbols, days_back=days_back, use_cache=use_cache)
                if av_articles:
                    return av_articles, "alphavantage"
            except Exception as err:
                logger.warning("Alpha Vantage secondary fallback failed: %s", err)

    # Case 2: US tickers with Alpha Vantage configured
    elif av_key:
        try:
            logger.info("Attempting Alpha Vantage news fetch for %s...", clean_symbols)
            articles = fetch_alpha_vantage_news(clean_symbols, days_back=days_back, use_cache=use_cache)
            if articles:
                return articles, "alphavantage"
            logger.info("Alpha Vantage returned 0 articles; falling back to Yahoo Finance for %s", clean_symbols)
        except Exception as exc:
            logger.warning("Alpha Vantage fetch failed (%s); falling back to Yahoo Finance for %s", exc, clean_symbols)

        # Fallback to Yahoo Finance
        yf_articles = fetch_yahoo_news(clean_symbols, days_back=days_back, use_cache=use_cache)
        if yf_articles:
            return yf_articles, "yahoo_finance"

    # Case 3: No Alpha Vantage key -> Direct Yahoo Finance
    else:
        logger.info("No Alpha Vantage key configured; fetching news for %s via Yahoo Finance...", clean_symbols)
        articles = fetch_yahoo_news(clean_symbols, days_back=days_back, use_cache=use_cache)
        if articles:
            return articles, "yahoo_finance"

    # Case 4: Final offline fallback
    from news.local_store import get_curated_fallback_news
    fallback: List[NewsArticle] = []
    for s in clean_symbols:
        fallback.extend(get_curated_fallback_news(s))
    if fallback:
        return fallback, "curated_offline"

    return [], "none"


def invalidate_all_news_caches() -> None:
    """Invalidate all news memory caches."""
    invalidate_av_cache()
    invalidate_yf_cache()
