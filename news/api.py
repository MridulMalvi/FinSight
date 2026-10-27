import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from news.ingest import load_news_csv
from news.features import aggregate_features, analyze_articles
from api.auth import require_user

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resolve_news_file(path_str: str) -> str:
    p = Path(path_str)
    if p.exists():
        return str(p)
    alt = PROJECT_ROOT / path_str
    if alt.exists():
        return str(alt)
    return path_str

router = APIRouter(prefix="/api/v1/news", tags=["news"])


class NewsFeatureRequest(BaseModel):
    csv_file: str = Field(default="data/raw/news/sample_news.csv", description="Path to news CSV file")
    symbol: str | None = Field(default="RELIANCE", description="Stock ticker symbol")

    model_config = {
        "json_schema_extra": {
            "example": {
                "csv_file": "data/raw/news/sample_news.csv",
                "symbol": "RELIANCE"
            }
        }
    }


class InlineArticle(BaseModel):
    title: str = Field(description="News headline")
    summary: Optional[str] = Field(default="", description="Optional article body")
    source: Optional[str] = Field(default="", description="Source label")


class AnalyzeTextRequest(BaseModel):
    articles: List[InlineArticle] = Field(description="List of news articles to analyze")
    symbol: Optional[str] = Field(default=None, description="Optional stock ticker for filtering")

    model_config = {
        "json_schema_extra": {
            "example": {
                "articles": [
                    {"title": "Reliance profit surges 20%", "summary": "Net profit rose on strong margins."},
                    {"title": "Oil prices fall on global slowdown fears", "source": "Reuters"}
                ],
                "symbol": "RELIANCE"
            }
        }
    }


@router.post("/analyze")
def analyze(request: NewsFeatureRequest, session: dict = Depends(require_user)):
    file_path = resolve_news_file(request.csv_file)
    if not Path(file_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"News CSV file not found: '{request.csv_file}'. Expected a CSV with title/headline and published_at/date columns."
        )
    try:
        articles = load_news_csv(file_path)
        df = analyze_articles(articles)
        records = [
            {k: (None if (isinstance(v, float) and v != v) else v) for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]
        return {"success": True, "articles": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/features")
def features(request: NewsFeatureRequest, session: dict = Depends(require_user)):
    file_path = resolve_news_file(request.csv_file)
    if not Path(file_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"News CSV file not found: '{request.csv_file}'. Expected a CSV with title/headline and published_at/date columns."
        )
    try:
        articles = load_news_csv(file_path)
        return {"success": True, "symbol": request.symbol, "features": aggregate_features(articles, request.symbol)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/analyze-text")
def analyze_text(request: AnalyzeTextRequest, session: dict = Depends(require_user)):
    """Analyze a list of inline news articles (no CSV needed). Returns per-article NLP results + aggregated features."""
    if not request.articles:
        raise HTTPException(status_code=400, detail="At least one article is required.")
    try:
        from news.models import NewsArticle
        news_articles = [
            NewsArticle(
                title=a.title,
                summary=a.summary or "",
                source=a.source or "",
                symbol=request.symbol,
            )
            for a in request.articles
        ]
        if news_articles:
            try:
                from news.alpha_vantage import save_today_news_files
                save_today_news_files(news_articles, [request.symbol or "MARKET"], provider="inline_analysis")
            except Exception:
                pass
        df = analyze_articles(news_articles)
        records = [
            {k: (None if (isinstance(v, float) and v != v) else v) for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]
        agg = aggregate_features(news_articles, symbol=request.symbol)
        return {
            "success": True,
            "articles": records,
            "aggregated_features": agg,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Relevance Scoring ────────────────────────────────────────────────────────

def _score_article_relevance(article_dict: dict, symbol: str) -> dict:
    """
    Score how relevant an article is to a specific company/symbol.

    Relevance is a composite [0.0, 1.0] score built from:
      1. Marketaux entity relevance score (direct API signal, 0–1)
      2. Symbol/ticker match in title or summary
      3. Company sector keyword match (sector_tags)
      4. Detected semantic themes that have non-zero company exposure

    Returns the article dict enriched with:
      - relevance_score: float [0, 1]
      - relevance_tier: "HIGH" | "MEDIUM" | "LOW" | "GENERAL"
      - relevance_reasons: list[str]  (human-readable explanation)
      - is_relevant: bool  (True if relevance_score >= 0.35)
      - matched_themes: list[str]
    """
    from news.company_context import get_profile, normalize_symbol
    from news.semantic import detect_themes

    canonical = normalize_symbol(symbol)
    profile = get_profile(canonical)
    sector_tags = [t.lower() for t in profile.get("sector_tags", [])]
    company_name = profile.get("name", "").lower()
    company_themes = profile.get("themes", {})

    title   = (article_dict.get("title") or "").lower()
    summary = (article_dict.get("summary") or "").lower()
    text    = title + " " + summary

    reasons: list[str] = []
    score: float = 0.0

    # 1. Marketaux entity relevance (most authoritative)
    mx_rel = article_dict.get("marketaux_relevance")
    if mx_rel is not None:
        mx_val = float(mx_rel)
        score += mx_val * 0.50           # up to +0.50
        if mx_val >= 0.70:
            reasons.append(f"Marketaux relevance {mx_val:.2f} — highly entity-matched")
        elif mx_val >= 0.40:
            reasons.append(f"Marketaux relevance {mx_val:.2f} — entity-matched")

    # 2. Article symbol tag matches the query symbol (or its canonical form)
    art_sym = (article_dict.get("symbol") or "").upper().split(".")[0]
    art_symbols = [s.upper().split(".")[0].replace("_NS", "").replace("_BO", "") for s in (article_dict.get("symbols") or [])]
    if art_sym:
        art_symbols.append(art_sym)

    clean_sym = symbol.upper().split(".")[0].replace("_HISTORICAL", "").replace("_HIST", "").replace("_NS", "").replace("_BO", "")
    clean_canon = canonical.upper().split(".")[0].replace("_NS", "").replace("_BO", "")
    allowed_symbols = {clean_sym, clean_canon, symbol.upper()}
    if "RELIANCE" in clean_sym or "RELIANCE" in clean_canon:
        allowed_symbols.add("RELIANCE")
    if "INFY" in clean_sym or "INFOSYS" in clean_sym or "INFY" in clean_canon:
        allowed_symbols.update(["INFY", "INFOSYS"])
    if "MSFT" in clean_sym or "MSFT" in clean_canon:
        allowed_symbols.update(["MSFT", "MICROSOFT"])
    if "AAPL" in clean_sym or "AAPL" in clean_canon:
        allowed_symbols.update(["AAPL", "APPLE"])
    if clean_canon in ("GOOGL", "GOOG"):
        allowed_symbols.update(["GOOGL", "GOOG", "GOOGLE", "ALPHABET"])
    if "NVDA" in clean_sym or "NVDA" in clean_canon:
        allowed_symbols.update(["NVDA", "NVIDIA"])
    if "TCS" in clean_sym or "TCS" in clean_canon:
        allowed_symbols.update(["TCS", "TATA"])
    if "TATAMOTORS" in clean_sym or "TATAMOTORS" in clean_canon:
        allowed_symbols.update(["TATAMOTORS", "TATA", "JLR"])
    if "HDFC" in clean_sym or "HDFC" in clean_canon:
        allowed_symbols.update(["HDFCBANK", "HDFC"])
    if "ICICI" in clean_sym or "ICICI" in clean_canon:
        allowed_symbols.update(["ICICIBANK", "ICICI"])
    if "TSLA" in clean_sym or "TSLA" in clean_canon:
        allowed_symbols.update(["TSLA", "TESLA"])

    matched_tag = next((s for s in art_symbols if s in allowed_symbols), None)
    if matched_tag:
        score += 0.30
        reasons.append(f"Article tagged to {matched_tag}")

    # 3. Company name / ticker in text
    ticker_candidates = [s.lower() for s in allowed_symbols if len(s) > 2]
    name_words = [w.lower() for w in company_name.split() if len(w) > 3 and w.lower() not in ("corporation", "ltd", "limited")]
    if any(tc in title for tc in ticker_candidates) or any(w in title for w in name_words):
        score += 0.35
        reasons.append(f"Company/ticker '{profile.get('name', symbol)}' featured in headline")
    elif any(tc in text for tc in ticker_candidates):
        score += 0.20
        reasons.append(f"Ticker '{ticker_candidates[0]}' found in text")
    elif name_words and any(w in text for w in name_words):
        score += 0.20
        reasons.append(f"Company name '{profile.get('name', '')}' found in text")

    # 4. Sector keyword match
    sector_hits = [tag for tag in sector_tags if tag in text]
    if sector_hits:
        hit_score = min(0.20, len(sector_hits) * 0.07)
        score += hit_score
        reasons.append(f"Sector keywords matched: {', '.join(sector_hits[:3])}")

    # 5. Semantic theme match with company exposure
    detected = detect_themes(text)
    company_theme_hits = [t for t in detected if abs(company_themes.get(t, 0.0)) >= 0.10]
    if company_theme_hits:
        theme_score = min(0.25, len(company_theme_hits) * 0.08)
        score += theme_score
        reasons.append(f"Macro themes relevant to {symbol}: {', '.join(company_theme_hits[:3])}")

    # Cap at 1.0
    score = min(1.0, score)

    # Tier assignment
    if score >= 0.65:
        tier = "HIGH"
    elif score >= 0.40:
        tier = "MEDIUM"
    elif score >= 0.20:
        tier = "LOW"
    else:
        tier = "GENERAL"

    return {
        **article_dict,
        "relevance_score":   round(score, 3),
        "relevance_tier":    tier,
        "relevance_reasons": reasons,
        "is_relevant":       score >= 0.35,
        "matched_themes":    company_theme_hits,
    }


# ─── Marketaux Live News Endpoints ────────────────────────────────────────────

@router.get("/live/{symbol}")
def live_news(
    symbol: str,
    days_back: int = Query(default=2, ge=1, le=7, description="Days of news history to fetch (1–7)"),
    use_cache: bool = Query(default=True, description="Whether to use cached results"),
    session: dict = Depends(require_user),
):
    """Fetch live financial news for a stock symbol from Marketaux API.

    Returns per-article NLP analysis enriched with:
      - Marketaux's own sentiment & relevance scores
      - Company-specific relevance scoring
      - Segregated 'relevant' vs 'general' article lists
    Articles are cached for 30 minutes to protect free-tier quota.
    """
    try:
        from news.provider import fetch_live_news
        articles, provider_name = fetch_live_news(symbols=[symbol], days_back=days_back, use_cache=use_cache)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        logger.warning("live_news fallback for %s: %s", symbol, exc)
        articles, provider_name = [], "offline"

    if not articles:
        return {
            "success": True,
            "symbol": symbol,
            "days_back": days_back,
            "article_count": 0,
            "relevant_count": 0,
            "general_count": 0,
            "articles": [],
            "relevant_articles": [],
            "general_articles": [],
            "message": "No news found for this symbol in the requested date range.",
        }

    # NLP analysis
    df = analyze_articles(articles)
    records = [
        {k: (None if (isinstance(v, float) and v != v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]

    # Score relevance for each article
    scored = [_score_article_relevance(r, symbol) for r in records]

    # Sort: relevant first (by relevance_score desc), then general
    relevant  = sorted([r for r in scored if r["is_relevant"]],
                       key=lambda x: x["relevance_score"], reverse=True)
    general   = sorted([r for r in scored if not r["is_relevant"]],
                       key=lambda x: x["relevance_score"], reverse=True)

    return {
        "success": True,
        "symbol": symbol,
        "days_back": days_back,
        "article_count": len(scored),
        "relevant_count": len(relevant),
        "general_count": len(general),
        "source": provider_name,
        # All articles (for backward-compatibility)
        "articles": scored,
        # Segregated lists used by the frontend panels
        "relevant_articles": relevant,
        "general_articles": general,
    }


@router.get("/live/{symbol}/features")
def live_news_features(
    symbol: str,
    days_back: int = Query(default=2, ge=1, le=7, description="Days of news history to fetch (1–7)"),
    use_cache: bool = Query(default=True, description="Whether to use cached results"),
    session: dict = Depends(require_user),
):
    """Fetch live news from Alpha Vantage for a symbol and return aggregated ML-ready features."""
    try:
        from news.alpha_vantage import fetch_alpha_vantage_news
        articles = fetch_alpha_vantage_news(symbols=[symbol], days_back=days_back, use_cache=use_cache)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        logger.warning("live_news_features error for %s: %s", symbol, exc)
        articles = []

    features = aggregate_features(articles, symbol=symbol)
    return {
        "success": True,
        "symbol": symbol,
        "days_back": days_back,
        "article_count": len(articles),
        "source": "alphavantage",
        "features": features,
    }


@router.delete("/live/cache")
def clear_live_cache(session: dict = Depends(require_user)):
    """Invalidate all cached news results."""
    try:
        from news.provider import invalidate_all_news_caches
        invalidate_all_news_caches()
        return {"success": True, "message": "All news memory caches cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
