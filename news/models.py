from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import hashlib

@dataclass
class NewsArticle:
    title: str
    summary: str = ""
    content: str = ""
    source: str = ""
    url: str = ""
    published_at: Optional[datetime] = None
    symbol: Optional[str] = None
    symbols: Optional[list[str]] = None
    sector: Optional[str] = None
    language: str = "en"
    # Marketaux API-provided scores (None when article came from CSV/RSS)
    marketaux_sentiment: Optional[float] = None   # range [-1, 1] from Marketaux
    marketaux_relevance: Optional[float] = None   # entity relevance score [0, 1]

    @property
    def text(self) -> str:
        return " ".join(x for x in [self.title, self.summary, self.content] if x).strip()

    @property
    def content_hash(self) -> str:
        raw = " ".join(self.title.lower().split())
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self):
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat() if self.published_at else None
        d["content_hash"] = self.content_hash
        return d
