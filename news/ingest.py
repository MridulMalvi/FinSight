import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, List, Optional

from news.models import NewsArticle

try:
    import feedparser
except ImportError:
    feedparser = None


def _parse_date(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value) if isinstance(value, str) else None
        if dt is not None:
            return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_news_csv(path: str) -> List[NewsArticle]:
    import pandas as pd
    df = pd.read_csv(path)
    aliases = {c.lower().strip(): c for c in df.columns}
    def col(*names, default=""):
        for n in names:
            if n in aliases:
                return aliases[n]
        return None
    title_col = col("title", "headline")
    if not title_col:
        raise ValueError("News CSV needs a title/headline column")
    summary_col = col("summary", "description")
    content_col = col("content", "article")
    source_col = col("source", "publisher")
    url_col = col("url", "link")
    date_col = col("published_at", "published", "date", "datetime")
    symbol_col = col("symbol", "ticker")
    sector_col = col("sector")
    out = []
    for _, r in df.iterrows():
        out.append(NewsArticle(
            title=str(r[title_col]),
            summary=str(r[summary_col]) if summary_col and pd.notna(r[summary_col]) else "",
            content=str(r[content_col]) if content_col and pd.notna(r[content_col]) else "",
            source=str(r[source_col]) if source_col and pd.notna(r[source_col]) else "",
            url=str(r[url_col]) if url_col and pd.notna(r[url_col]) else "",
            published_at=_parse_date(r[date_col]) if date_col and pd.notna(r[date_col]) else None,
            symbol=str(r[symbol_col]).upper() if symbol_col and pd.notna(r[symbol_col]) else None,
            sector=str(r[sector_col]) if sector_col and pd.notna(r[sector_col]) else None,
        ))
    return deduplicate_articles(out)


def fetch_rss(urls: Iterable[str], timeout: int = 15) -> List[NewsArticle]:
    """Fetch permitted RSS feeds. For commercial use, confirm provider licensing/terms."""
    if feedparser is None:
        raise ImportError("feedparser is required for RSS ingestion")
    articles = []
    for url in urls:
        feed = feedparser.parse(url)
        source = getattr(feed.feed, "title", "RSS")
        for e in feed.entries:
            title = getattr(e, "title", "").strip()
            if not title:
                continue
            published = getattr(e, "published", None) or getattr(e, "updated", None)
            articles.append(NewsArticle(
                title=title,
                summary=getattr(e, "summary", ""),
                content=getattr(e, "description", ""),
                source=source,
                url=getattr(e, "link", ""),
                published_at=_parse_date(published),
            ))
    return deduplicate_articles(articles)


def deduplicate_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
    seen = set()
    result = []
    for a in sorted(articles, key=lambda x: x.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True):
        # Hash title + normalized key phrases; title hash catches syndicated duplicates.
        key = a.content_hash
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result
