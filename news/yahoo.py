"""Yahoo Finance and Live Financial News Engine.

Fetches real-time financial news articles for global and Indian stocks
(e.g., TCS.NS, RELIANCE.NS, NVDA, AAPL) using live financial news feeds
and Yahoo Finance API.
Enforces strict date filtering so 'Today only' returns strictly today's news.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple

import yfinance as yf

from news.models import NewsArticle
from news.company_context import get_profile, normalize_symbol

logger = logging.getLogger(__name__)

# ─── In-Memory Cache ──────────────────────────────────────────────────────────
_YF_CACHE: Dict[str, Tuple[float, List[NewsArticle]]] = {}  # key → (expires_ts, articles)
_YF_CACHE_TTL_SECONDS = 900  # 15 minutes


def _yf_cache_key(symbols: List[str], days_back: int) -> str:
    return f"yf:{','.join(sorted(s.upper() for s in symbols))}|days={days_back}"


def _get_yf_cached(key: str) -> Optional[List[NewsArticle]]:
    entry = _YF_CACHE.get(key)
    if entry and time.monotonic() < entry[0]:
        return entry[1]
    return None


def _set_yf_cache(key: str, articles: List[NewsArticle]) -> None:
    _YF_CACHE[key] = (time.monotonic() + _YF_CACHE_TTL_SECONDS, articles)


def invalidate_yf_cache() -> None:
    """Clear Yahoo Finance in-memory cache."""
    _YF_CACHE.clear()


def build_company_query(symbol: str) -> str:
    """Build high-precision financial search query for a ticker."""
    canonical = normalize_symbol(symbol)
    profile = get_profile(canonical)
    raw_name = profile.get("name", "").strip()

    clean_sym = (
        symbol.upper()
        .split(".")[0]
        .replace("_HISTORICAL", "")
        .replace("_HIST", "")
        .replace("_NS", "")
        .replace("_BO", "")
    )

    # Special high-precision search queries for the 10 target companies
    if canonical == "GOLDBEES":
        return '"Gold price" OR "Gold BeES" OR "Gold ETF"'
    if canonical == "MID150BEES":
        return '"Nifty Midcap 150" OR "Midcap 150" OR "Midcap ETF"'
    if canonical == "POWERGRID":
        return 'POWERGRID stock OR "Power Grid" OR "PGCIL"'
    if canonical == "IOC":
        return 'IOC stock OR "Indian Oil" OR "IOCL"'
    if canonical == "IRFC":
        return 'IRFC stock OR "Indian Railway Finance"'
    if canonical == "HUDCO":
        return 'HUDCO stock OR "Housing and Urban Development Corporation"'
    if canonical == "BEL":
        return 'BEL stock OR "Bharat Electronics"'
    if canonical == "ASHOKLEY":
        return 'ASHOKLEY stock OR "Ashok Leyland"'
    if canonical == "SUZLON":
        return 'SUZLON stock OR "Suzlon Energy"'
    if canonical == "IDBI":
        return 'IDBI stock OR "IDBI Bank"'

    # Filter out corporate noise words
    words = [
        w for w in raw_name.split()
        if w.lower() not in ("ltd", "limited", "corp", "corporation", "inc", "co", "plc")
    ]
    clean_name = " ".join(words)

    if clean_name and clean_sym and clean_name.upper() != clean_sym:
        return f'{clean_sym} stock OR "{clean_name}"'
    elif clean_name:
        return f'"{clean_name}" stock'
    else:
        return f'{clean_sym} stock'



# Month name → number map for parsing description dates
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Matches patterns like "March 31, 2026 4:18 PM" or "Sep 24, 2026"
_DESC_DATE_RE = re.compile(
    r'\b(?P<mon>[A-Za-z]{3,9})\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})',
    re.IGNORECASE,
)


def _extract_desc_date(description: str) -> Optional[datetime]:
    """Parse the true article publication date from a Google News RSS description.

    Google News descriptions often end with: '\u2014 Source Name | March 31, 2026 4:18 PM'
    We extract this date and return it as a UTC-anchored datetime so it can
    be used to filter out re-surfaced old articles.
    """
    if not description:
        return None
    # Only look in the last 100 chars where Google usually puts the date
    text = description[-120:] if len(description) > 120 else description
    m = _DESC_DATE_RE.search(text)
    if not m:
        return None
    try:
        month = _MONTH_MAP.get(m.group("mon").lower())
        if not month:
            return None
        day = int(m.group("day"))
        year = int(m.group("year"))
        if year < 2000 or year > 2100:
            return None
        return datetime(year, month, day, tzinfo=timezone.utc)
    except Exception:
        return None


def fetch_rss_articles(query: str, cutoff: datetime, symbol: str) -> List[NewsArticle]:
    """Fetch real-time financial news from Google Financial RSS."""
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    clean_sym = symbol.upper().split(".")[0].replace("_HISTORICAL", "").replace("_HIST", "").replace("_NS", "").replace("_BO", "")

    articles: List[NewsArticle] = []
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            root = ET.fromstring(resp.read())
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                if not title:
                    continue

                link = (item.findtext("link") or "").strip()
                pub_str = item.findtext("pubDate")
                source = (item.findtext("source") or "").strip() or "Financial News"
                desc = (item.findtext("description") or "").strip()
                summary = re.sub(r'<[^>]+>', '', desc).strip()

                pub_dt = None
                if pub_str:
                    try:
                        pub_dt = parsedate_to_datetime(pub_str)
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass

                # ── Google News re-surface guard ──────────────────────────────
                # Google News RSS sometimes assigns today's crawl date to old
                # articles. Extract the REAL publication date from the raw description
                # (e.g. "News On AIR | March 31, 2026 4:18 PM") — use raw desc
                # BEFORE HTML stripping since Google embeds date inside HTML tags.
                desc_date = _extract_desc_date(desc)  # raw HTML description
                if desc_date is not None and pub_dt is not None:
                    if desc_date < pub_dt - timedelta(days=1):
                        # Article is significantly older than its RSS crawl date
                        logger.info(
                            "Re-surface detected: RSS pubDate=%s but article date=%s — skipping '%s'",
                            pub_dt.date(), desc_date.date(), title[:60]
                        )
                        pub_dt = desc_date  # Use the true article date

                # Strict cutoff: do not include if older than cutoff
                if pub_dt and pub_dt < cutoff:
                    continue
                if pub_dt is None:
                    continue  # Skip articles with no parseable date

                art = NewsArticle(
                    title=title,
                    summary=summary,
                    content="",
                    source=source,
                    url=link,
                    published_at=pub_dt,
                    symbol=symbol.upper(),
                    symbols=[symbol.upper(), clean_sym],
                    marketaux_relevance=0.90,
                )
                articles.append(art)
    except Exception as exc:
        logger.warning("Live RSS fetch failed for query '%s': %s", query, exc)

    return articles


def parse_yf_news_item(item: dict, target_symbol: str, cutoff: datetime) -> Optional[NewsArticle]:
    """Parse a single raw Yahoo Finance news item into a NewsArticle."""
    content = item.get("content")
    if isinstance(content, dict):
        title = (content.get("title") or "").strip()
        summary = (content.get("summary") or content.get("description") or "").strip()

        source = ""
        prov = content.get("provider")
        if isinstance(prov, dict):
            source = prov.get("displayName") or prov.get("sourceId") or ""
        if not source:
            source = item.get("publisher") or "Yahoo Finance"

        url = ""
        c_url = content.get("canonicalUrl")
        if isinstance(c_url, dict):
            url = c_url.get("url") or ""
        if not url:
            ct_url = content.get("clickThroughUrl")
            if isinstance(ct_url, dict):
                url = ct_url.get("url") or ""
        if not url:
            url = item.get("link") or ""

        pub_dt = None
        pub_str = content.get("pubDate") or content.get("displayTime")
        if pub_str:
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
    else:
        title = (item.get("title") or "").strip()
        summary = (item.get("summary") or "").strip()
        source = item.get("publisher") or "Yahoo Finance"
        url = item.get("link") or ""
        pub_dt = None
        pub_sec = item.get("providerPublishTime")
        if pub_sec:
            try:
                pub_dt = datetime.fromtimestamp(pub_sec, tz=timezone.utc)
            except Exception:
                pass

    if not title:
        return None

    # Strict cutoff
    if pub_dt and pub_dt < cutoff:
        return None

    norm_sym = target_symbol.strip().upper()
    clean_sym = norm_sym.split(".")[0].replace("_HISTORICAL", "").replace("_HIST", "")
    symbols = [norm_sym]
    if clean_sym != norm_sym:
        symbols.append(clean_sym)

    return NewsArticle(
        title=title,
        summary=summary,
        content="",
        source=source or "Yahoo Finance",
        url=url,
        published_at=pub_dt,
        symbol=norm_sym,
        symbols=symbols,
        marketaux_relevance=0.90,
    )


def fetch_yahoo_news(
    symbols: List[str],
    days_back: int = 1,
    use_cache: bool = True,
    limit: int = 30,
) -> List[NewsArticle]:
    """Fetch live news articles with strict date window enforcement.

    Combines real-time financial RSS feeds and Yahoo Finance news,
    ensuring that 'Today only' (days_back=1) strictly returns articles
    published within the last 24 hours.

    Args:
        symbols: Stock ticker symbols (e.g. ['TCS.NS'], ['NVDA']).
        days_back: Maximum days back for news articles (1 for today).
        use_cache: Whether to use 15-minute memory cache.
        limit: Max articles to return.
    """
    if not symbols:
        return []

    cache_key = _yf_cache_key(symbols, days_back)
    if use_cache:
        cached = _get_yf_cached(cache_key)
        if cached is not None:
            logger.info("Live News Engine: serving %d articles from memory cache for %s (days_back=%d)", len(cached), symbols, days_back)
            return cached

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, days_back))

    all_articles: List[NewsArticle] = []
    seen_hashes: set[str] = set()

    for sym in symbols:
        clean_sym = sym.strip()
        # 1. Fetch real-time RSS news
        query = build_company_query(clean_sym)
        rss_articles = fetch_rss_articles(query, cutoff=cutoff, symbol=clean_sym)
        for a in rss_articles:
            if a.content_hash not in seen_hashes:
                seen_hashes.add(a.content_hash)
                all_articles.append(a)

        # 2. Fetch Yahoo Finance ticker news
        try:
            ticker = yf.Ticker(clean_sym)
            raw_items = ticker.news or []
            for it in raw_items:
                art = parse_yf_news_item(it, target_symbol=clean_sym, cutoff=cutoff)
                if art and art.content_hash not in seen_hashes:
                    seen_hashes.add(art.content_hash)
                    all_articles.append(art)
        except Exception as exc:
            logger.warning("Yahoo Finance ticker news fetch failed for %s: %s", clean_sym, exc)

    # Strict sorting: newest published first
    all_articles.sort(
        key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    final_articles = all_articles[:limit]

    if final_articles:
        if use_cache:
            _set_yf_cache(cache_key, final_articles)
        try:
            from news.alpha_vantage import save_today_news_files
            save_today_news_files(final_articles, target_symbols=symbols, provider="yahoo_finance")
        except Exception as err:
            logger.warning("Failed to save today_news file: %s", err)

    logger.info("Live News Engine: retrieved %d articles for %s within last %d day(s)", len(final_articles), symbols, days_back)
    return final_articles
