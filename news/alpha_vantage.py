"""Alpha Vantage News & Sentiment API client.

Fetches current financial news articles mapped to stock tickers with
Alpha Vantage's machine-learning-derived sentiment and relevance scores.

API Endpoint:
    https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={TICKER}&time_from={YYYYMMDDTHHMM}&limit=50&apikey={KEY}

Free Tier:
    - 25 requests/day, 5 requests/minute
    - Up to 50 articles per request (unlike Marketaux's 3-article free cap)
    - Results cached in-memory (30 min) and on disk in today_news.json/md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

from news.models import NewsArticle

logger = logging.getLogger(__name__)

# ─── File & Memory Cache Paths ────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "news_cache"
_TODAY_NEWS_JSON = _CACHE_DIR / "today_news.json"
_TODAY_NEWS_MD = _CACHE_DIR / "today_news.md"

# ─── In-Memory Cache ──────────────────────────────────────────────────────────
_CACHE: Dict[str, Tuple[float, List[NewsArticle]]] = {}  # key → (expires_ts, articles)
_CACHE_TTL_SECONDS = 1800  # 30 minutes

_AV_BASE_URL = "https://www.alphavantage.co/query"


def _cache_key(symbols: List[str], days_back: int) -> str:
    return f"{','.join(sorted(s.upper() for s in symbols))}|days={days_back}"


def _get_cached(key: str) -> Optional[List[NewsArticle]]:
    entry = _CACHE.get(key)
    if entry and time.monotonic() < entry[0]:
        return entry[1]
    return None


def _set_cache(key: str, articles: List[NewsArticle]) -> None:
    _CACHE[key] = (time.monotonic() + _CACHE_TTL_SECONDS, articles)


def _clear_expired_cache() -> None:
    now = time.monotonic()
    expired = [k for k, (exp, _) in _CACHE.items() if now >= exp]
    for k in expired:
        del _CACHE[k]


# ─── Disk Storage for Present Day News ───────────────────────────────────────

def save_today_news_files(
    articles: List[NewsArticle],
    target_symbols: List[str],
    provider: str = "alphavantage",
) -> None:
    """Save articles to data/news_cache/today_news.json and today_news.md strictly for today.

    Includes ticker summary breakdown and bullish/bearish statistics in today_news.md.
    """
    if not articles:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now().strftime("%Y-%m-%d")
        symbol_tag = ",".join(target_symbols)

        existing_records: List[dict] = []
        if _TODAY_NEWS_JSON.exists():
            try:
                with open(_TODAY_NEWS_JSON, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, list):
                        existing_records = [
                            item for item in raw_data
                            if item.get("fetched_date") == today_str
                        ]
            except Exception as err:
                logger.warning("Could not parse existing today_news.json: %s", err)
                existing_records = []

        seen_hashes = {r.get("content_hash") for r in existing_records if r.get("content_hash")}

        new_count = 0
        for a in articles:
            # Skip articles with no publication date — can't verify freshness
            if not a.published_at:
                continue
            if a.content_hash not in seen_hashes:
                seen_hashes.add(a.content_hash)
                record = a.to_dict()
                record["fetched_date"] = today_str
                record["queried_symbol"] = symbol_tag
                record["provider"] = provider
                existing_records.append(record)
                new_count += 1

        existing_records.sort(
            key=lambda r: r.get("published_at") or "",
            reverse=True,
        )

        with open(_TODAY_NEWS_JSON, "w", encoding="utf-8") as f:
            json.dump(existing_records, f, indent=2, ensure_ascii=False)

        # Compute ticker breakdown and sentiment breakdown
        ticker_counts: Dict[str, int] = {}
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for item in existing_records:
            sym = (item.get("symbol") or item.get("queried_symbol") or "MARKET").upper()
            ticker_counts[sym] = ticker_counts.get(sym, 0) + 1
            sentiment_score = item.get("marketaux_sentiment")
            if sentiment_score is not None:
                if sentiment_score > 0.05:
                    bullish_count += 1
                elif sentiment_score < -0.05:
                    bearish_count += 1
                else:
                    neutral_count += 1
            else:
                neutral_count += 1

        ticker_summary_str = " | ".join(f"**{tk}**: {cnt}" for tk, cnt in sorted(ticker_counts.items()))

        md_lines = [
            f"# 📰 Today's Stock News Cache ({today_str})",
            f"*Auto-updated by FinSight AI News Engine — Provider: Alpha Vantage & Live Engine*",
            "",
            "### 📊 Today's Overview",
            f"- **Total Articles**: `{len(existing_records)}`",
            f"- **Tracked Tickers**: {ticker_summary_str}",
            f"- **Market Sentiment Breakdown**: 🟢 `{bullish_count} Bullish` | 🔴 `{bearish_count} Bearish` | ⚪ `{neutral_count} Neutral`",
            f"- **Last Updated**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "---",
            "",
        ]

        for i, item in enumerate(existing_records, 1):
            sym = item.get("symbol") or item.get("queried_symbol") or "MARKET"
            sentiment_score = item.get("marketaux_sentiment")
            if sentiment_score is not None:
                if sentiment_score > 0.05:
                    sent_badge = f"🟢 **Bullish (+{sentiment_score:.2f})**"
                elif sentiment_score < -0.05:
                    sent_badge = f"🔴 **Bearish ({sentiment_score:.2f})**"
                else:
                    sent_badge = f"⚪ **Neutral ({sentiment_score:.2f})**"
            else:
                sent_badge = "⚪ **Neutral / Unscored**"

            pub_str = item.get("published_at") or "Unknown date"
            source = item.get("source") or "Unknown source"
            title = item.get("title") or "No title"
            summary = item.get("summary") or item.get("content") or "No summary provided."
            url = item.get("url")

            md_lines.append(f"### {i}. [{sym}] {title}")
            md_lines.append(f"- **Source**: {source} | **Published**: `{pub_str}`")
            md_lines.append(f"- **Sentiment**: {sent_badge}")
            if url:
                md_lines.append(f"- **Article Link**: [{url}]({url})")
            md_lines.append(f"\n> {summary.strip()}\n")
            md_lines.append("---")
            md_lines.append("")

        with open(_TODAY_NEWS_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(
            "Saved %d articles to %s and %s (total today: %d)",
            new_count, _TODAY_NEWS_JSON.name, _TODAY_NEWS_MD.name, len(existing_records),
        )
    except Exception as exc:
        logger.error("Failed to save today's news file: %s", exc, exc_info=True)


# Backwards compatibility alias
_save_today_news_files = save_today_news_files


def _load_today_news_file(symbols: List[str], days_back: Optional[int] = None) -> Optional[List[NewsArticle]]:
    """Load articles from today_news.json if present day records match the symbol and date cutoff."""
    if not _TODAY_NEWS_JSON.exists():
        return None
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        now_utc = datetime.now(timezone.utc)
        cutoff = (now_utc - timedelta(days=max(1, days_back))) if days_back is not None else None

        with open(_TODAY_NEWS_JSON, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            return None

        target_tags = set()
        for s in symbols:
            norm = normalize_to_av(s).upper()
            target_tags.add(norm)
            target_tags.add(norm.split(".")[0])
            target_tags.add(s.upper())

        matched_articles = []
        for item in records:
            if item.get("fetched_date") != today_str:
                continue

            item_sym = (item.get("symbol") or "").upper()
            q_sym = (item.get("queried_symbol") or "").upper()
            item_syms = [s.upper() for s in (item.get("symbols") or [])]

            if (
                item_sym in target_tags
                or q_sym in target_tags
                or any(s in target_tags for s in item_syms)
                or any(t in q_sym for t in target_tags)
            ):
                pub_dt = None
                if item.get("published_at"):
                    try:
                        pub_dt = datetime.fromisoformat(item["published_at"])
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        pub_dt = None

                # Skip articles with no date — can't verify freshness
                if pub_dt is None:
                    continue

                # Enforce strict cutoff check: do not serve articles older than requested window
                if cutoff and pub_dt < cutoff:
                    continue

                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("summary", ""),
                    content=item.get("content", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    published_at=pub_dt,
                    symbol=item.get("symbol"),
                    symbols=item.get("symbols"),
                    sector=item.get("sector"),
                    language=item.get("language", "en"),
                    marketaux_sentiment=item.get("marketaux_sentiment"),
                    marketaux_relevance=item.get("marketaux_relevance"),
                )
                matched_articles.append(article)

        if matched_articles:
            logger.info("Loaded %d articles from %s for %s", len(matched_articles), _TODAY_NEWS_JSON.name, symbols)
            return matched_articles
    except Exception as exc:
        logger.warning("Failed to load today_news.json: %s", exc)
    return None


# ─── Symbol Normalization ─────────────────────────────────────────────────────

def normalize_to_av(symbol: str) -> str:
    """Normalize ticker symbol for Alpha Vantage News API.

    Alpha Vantage enforces: 'Ticker can only contain alphanumeric characters, colons, underscores, and hyphens'.
    Periods ('.') are strictly forbidden.

    Examples:
        RELIANCE / Reliance_historical → RELIANCE
        INFY / Infosys_historical      → INFY (NYSE ADR / global symbol)
        MSFT                           → MSFT
        AAPL                           → AAPL
    """
    s = symbol.strip().upper()
    # Strip suffixes and periods (Alpha Vantage rejects any dots)
    s_clean = (
        s.replace("_HISTORICAL", "")
        .replace("_HIST", "")
        .replace(".NS", "")
        .replace(".BO", "")
        .replace(".BSE", "")
        .replace("_NS", "")
        .replace("_BO", "")
        .replace(".", "")
    )

    if "INFOSYS" in s_clean or s_clean == "INFY":
        return "INFY"
    if "RELIANCE" in s_clean:
        return "RELIANCE"
    if "TCS" in s_clean:
        return "TCS"
    return s_clean


def _time_from_string(days_back: int = 2) -> str:
    """Return Alpha Vantage formatted time_from string (YYYYMMDDTHHMM).

    On Sunday and Monday, automatically extends window to 4 days to include Friday.
    """
    now = datetime.now(timezone.utc)
    effective_days = days_back
    if days_back == 2 and now.weekday() in (0, 6):
        effective_days = 4
    dt = now - timedelta(days=effective_days)
    return dt.strftime("%Y%m%dT%H%M")


def _parse_av_time(time_str: str) -> Optional[datetime]:
    """Parse Alpha Vantage time string '20231024T180000' to datetime."""
    if not time_str:
        return None
    try:
        # Standard format: YYYYMMDDTHHMMSS
        clean = time_str.replace("Z", "")
        return datetime.strptime(clean, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.strptime(time_str[:13], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        except Exception:
            return None


def _parse_av_article(feed_item: dict, target_tickers: List[str]) -> Optional[NewsArticle]:
    """Convert an Alpha Vantage feed item into a NewsArticle."""
    title = (feed_item.get("title") or "").strip()
    if not title:
        return None

    published_at = _parse_av_time(feed_item.get("time_published", ""))

    overall_sentiment = feed_item.get("overall_sentiment_score")
    try:
        overall_sentiment = float(overall_sentiment) if overall_sentiment is not None else None
    except (ValueError, TypeError):
        overall_sentiment = None

    ticker_sentiments = feed_item.get("ticker_sentiment", [])
    all_symbols = [t.get("ticker", "").upper() for t in ticker_sentiments if t.get("ticker")]

    # Find the most relevant matching ticker in ticker_sentiment list
    target_clean = {t.upper().split(".")[0] for t in target_tickers}
    target_clean.update({t.upper() for t in target_tickers})

    best_sentiment = overall_sentiment
    best_relevance = None
    primary_symbol = target_tickers[0] if target_tickers else None

    for ts in ticker_sentiments:
        tk = (ts.get("ticker") or "").upper()
        tk_base = tk.split(".")[0]
        if tk in target_clean or tk_base in target_clean:
            try:
                score = float(ts.get("ticker_sentiment_score"))
                rel = float(ts.get("relevance_score"))
                best_sentiment = score
                best_relevance = rel
                primary_symbol = tk
                break
            except (ValueError, TypeError):
                pass

    return NewsArticle(
        title=title,
        summary=feed_item.get("summary", ""),
        content="",
        source=feed_item.get("source", "") or feed_item.get("source_domain", ""),
        url=feed_item.get("url", ""),
        published_at=published_at,
        symbol=primary_symbol,
        symbols=all_symbols,
        marketaux_sentiment=best_sentiment,
        marketaux_relevance=best_relevance,
    )


def fetch_alpha_vantage_news(
    symbols: List[str],
    api_key: str = "",
    days_back: int = 2,
    limit: int = 50,
    use_cache: bool = True,
) -> List[NewsArticle]:
    """Fetch financial news and sentiment from Alpha Vantage.

    Args:
        symbols: Stock symbols (e.g. ['MSFT', 'INFY', 'RELIANCE']).
        api_key: Alpha Vantage API token. Falls back to env ALPHA_VANTAGE_API_KEY.
        days_back: Lookback window in days.
        limit: Number of articles to retrieve (default 50, max 1000).
        use_cache: Whether to use 30-min memory / today's disk cache.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    resolved_key = (
        api_key
        or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        or os.getenv("ALPHAVANTAGE_API_KEY", "")
    )

    # Clean ticker symbols for Alpha Vantage
    av_symbols = [normalize_to_av(s) for s in symbols]
    cache_key = _cache_key(av_symbols, days_back)

    if use_cache:
        _clear_expired_cache()
        cached = _get_cached(cache_key)
        if cached is not None:
            logger.info("Alpha Vantage: serving %d articles from memory cache (%s)", len(cached), cache_key)
            return cached

        disk_cached = _load_today_news_file(symbols, days_back=days_back)
        if disk_cached:
            logger.info("Alpha Vantage: serving %d articles from today's disk cache for %s", len(disk_cached), symbols)
            _set_cache(cache_key, disk_cached)
            return disk_cached

    if not resolved_key:
        # Check if demo key works for demo tickers
        if any(s.upper() in ("AAPL", "MSFT", "IBM") for s in av_symbols):
            resolved_key = "demo"
            logger.info("No ALPHA_VANTAGE_API_KEY set; falling back to demo key for %s", av_symbols)
        else:
            raise PermissionError(
                "Alpha Vantage API key not found. "
                "Add ALPHA_VANTAGE_API_KEY=<your_key> to your .env file and restart."
            )

    time_from = _time_from_string(days_back)
    tickers_param = ",".join(av_symbols)

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers_param,
        "time_from": time_from,
        "limit": min(limit, 100),
        "apikey": resolved_key,
    }

    try:
        resp = requests.get(_AV_BASE_URL, params=params, timeout=20)
    except Exception as exc:
        logger.warning("Alpha Vantage request failed: %s; checking offline cache/fallback", exc)
        disk_cached = _load_today_news_file(symbols, days_back=days_back)
        if disk_cached:
            return disk_cached
        from news.local_store import get_curated_fallback_news
        fallback_articles: List[NewsArticle] = []
        for s in symbols:
            fallback_articles.extend(get_curated_fallback_news(s))
        if fallback_articles:
            save_today_news_files(fallback_articles, target_symbols=symbols, provider="network_fallback")
            return fallback_articles
        return []

    if resp.status_code != 200:
        logger.warning("Alpha Vantage returned HTTP %s: %s", resp.status_code, resp.text)
        disk_cached = _load_today_news_file(symbols, days_back=days_back)
        if disk_cached:
            return disk_cached
        from news.local_store import get_curated_fallback_news
        fallback_articles: List[NewsArticle] = []
        for s in symbols:
            fallback_articles.extend(get_curated_fallback_news(s))
        if fallback_articles:
            save_today_news_files(fallback_articles, target_symbols=symbols, provider="http_error_fallback")
            return fallback_articles
        return []

    data = resp.json()

    # Check for API errors / note messages
    if "Error Message" in data:
        logger.warning("Alpha Vantage error: %s", data["Error Message"])
        disk_cached = _load_today_news_file(symbols, days_back=days_back)
        if disk_cached:
            return disk_cached
        from news.local_store import get_curated_fallback_news
        fallback_articles: List[NewsArticle] = []
        for s in symbols:
            fallback_articles.extend(get_curated_fallback_news(s))
        if fallback_articles:
            save_today_news_files(fallback_articles, target_symbols=symbols, provider="api_error_fallback")
            return fallback_articles
        return []

    if "Information" in data:
        # Rate limit info message
        msg = data["Information"]
        logger.warning("Alpha Vantage information/limit: %s", msg)
        # 1. If rate limited but disk cache exists, return disk cache
        disk_cached = _load_today_news_file(symbols, days_back=days_back)
        if disk_cached:
            logger.info("Alpha Vantage rate limit: serving %d articles from today_news disk cache for %s", len(disk_cached), symbols)
            return disk_cached

        # 2. Try curated offline fallback news (e.g. Reliance, Infosys)
        from news.local_store import get_curated_fallback_news
        fallback_articles: List[NewsArticle] = []
        for s in symbols:
            fallback_articles.extend(get_curated_fallback_news(s))

        if fallback_articles:
            logger.info("Alpha Vantage rate limit: loaded %d curated fallback articles for %s", len(fallback_articles), symbols)
            save_today_news_files(fallback_articles, target_symbols=symbols, provider="rate_limit_fallback")
            if use_cache:
                _set_cache(cache_key, fallback_articles)
            return fallback_articles

        logger.info("Alpha Vantage rate limit reached; returning empty list gracefully for %s", symbols)
        return []

    feed = data.get("feed", [])
    # If strict time_from produced 0 articles, fetch latest available articles for this symbol
    if not feed and "time_from" in params:
        logger.info("Alpha Vantage: 0 articles with time_from; fetching latest available news for %s", av_symbols)
        params_retry = dict(params)
        del params_retry["time_from"]
        try:
            resp_retry = requests.get(_AV_BASE_URL, params=params_retry, timeout=20)
            if resp_retry.status_code == 200:
                feed = resp_retry.json().get("feed", [])
        except Exception as err:
            logger.warning("Alpha Vantage retry without time_from failed: %s", err)
    articles: List[NewsArticle] = []
    seen_hashes: set[str] = set()

    for item in feed:
        article = _parse_av_article(item, target_tickers=av_symbols)
        if article and article.content_hash not in seen_hashes:
            seen_hashes.add(article.content_hash)
            articles.append(article)

    # Sort newest first
    articles.sort(
        key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    if articles:
        if use_cache:
            _set_cache(cache_key, articles)
        save_today_news_files(articles, target_symbols=symbols, provider="alphavantage")
    else:
        # Fallback to curated news if external API has 0 articles for the requested symbol
        from news.local_store import get_curated_fallback_news
        fallback_articles: List[NewsArticle] = []
        for s in symbols:
            fallback_articles.extend(get_curated_fallback_news(s))
        if fallback_articles:
            logger.info("Alpha Vantage returned 0 articles for %s; serving %d curated fallback articles", symbols, len(fallback_articles))
            save_today_news_files(fallback_articles, target_symbols=symbols, provider="market_fallback")
            if use_cache:
                _set_cache(cache_key, fallback_articles)
            return fallback_articles

    logger.info("Alpha Vantage: retrieved %d articles for %s", len(articles), av_symbols)
    return articles


def invalidate_av_cache() -> None:
    """Clear memory cache."""
    _CACHE.clear()
