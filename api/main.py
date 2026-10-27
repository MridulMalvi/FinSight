import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from ingestion.csv_provider import CSVMarketDataProvider
from ingestion.yfinance_provider import YFinanceMarketDataProvider, validate_yf_ticker, normalize_to_yf, get_yf_ticker_info
from ml.predict import predict, predict_multi_horizon
from news.api import router as news_router
from news.semantic import analyze_company_impact
from news.company_context import get_profile, list_symbols as list_known_symbols, normalize_symbol, get_currency
from api.auth import router as auth_router, get_current_user, require_user
from api.admin import router as admin_router, log_activity, check_analysis_quota
from database.connection import get_db, init_db
from database.models import AllowedSymbol, UserActivity
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

def resolve_file(path_str: str | None) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return str(p)
    alt = PROJECT_ROOT / path_str
    if alt.exists():
        return str(alt)
    return path_str


def resolve_price_csv(symbol: str) -> Optional[Path]:
    """Resolve price CSV path handling common ticker aliases (e.g. INFY -> Infosys_historical)."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    sym_clean = (
        symbol.strip().upper()
        .replace("_HISTORICAL", "")
        .replace("_HIST", "")
        .replace(".NS", "")
        .replace(".BO", "")
        .replace(".BSE", "")
        .replace(".", "")
    )

    candidates = [
        raw_dir / f"{symbol}.csv",
        raw_dir / f"{sym_clean}.csv",
        raw_dir / f"{sym_clean}_historical.csv",
        raw_dir / f"{sym_clean.capitalize()}_historical.csv",
    ]
    if "INFY" in sym_clean or "INFOSYS" in sym_clean:
        candidates.extend([
            raw_dir / "Infosys_historical.csv",
            raw_dir / "INFY.csv",
        ])
    if "RELIANCE" in sym_clean:
        candidates.extend([
            raw_dir / "Reliance_historical.csv",
            raw_dir / "RELIANCE.csv",
        ])

    for c in candidates:
        if c.exists():
            return c

    for f in raw_dir.glob("*.csv"):
        f_upper = f.stem.upper()
        if sym_clean in f_upper or f_upper.startswith(sym_clean):
            return f

    return None

app = FastAPI(title="FinSight AI Prediction API", version="0.4.0",
              description="Research MVP for probabilistic stock-direction predictions with News Intelligence, Auth, and Admin Panel.")

# Allow the frontend (any origin during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(news_router)
app.include_router(auth_router)
app.include_router(admin_router)

@app.on_event("startup")
def startup_event():
    """Initialize database tables and seed defaults on app start."""
    try:
        init_db()
        logger.info("[Startup] Database initialized successfully.")
    except Exception as e:
        logger.error("[Startup] DB init failed: %s", e)


class PredictionRequest(BaseModel):
    csv_file: str = Field(default="data/raw/RELIANCE.csv", description="Path to price OHLCV CSV file")
    news_csv_file: str | None = Field(default="data/raw/news/sample_news.csv", description="Optional path to news CSV")
    symbol: str | None = Field(default="RELIANCE", description="Stock ticker symbol")

    model_config = {
        "json_schema_extra": {
            "example": {
                "csv_file": "data/raw/RELIANCE.csv",
                "news_csv_file": "data/raw/news/sample_news.csv",
                "symbol": "RELIANCE"
            }
        }
    }


class InlineNewsItem(BaseModel):
    title: str = Field(description="News headline")
    summary: Optional[str] = Field(default="", description="Optional article body / summary")
    source: Optional[str] = Field(default="", description="Source / publisher label")
    url: Optional[str] = Field(default="", description="Original article URL (for Read button)")


class PredictWithNewsRequest(BaseModel):
    symbol: str = Field(default="RELIANCE", description="Stock ticker symbol — must match a CSV in data/raw/")
    news_items: List[InlineNewsItem] = Field(default=[], description="List of news articles to incorporate")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "RELIANCE",
                "news_items": [
                    {"title": "Reliance Q3 profit surges 20%", "summary": "Net profit rose sharply on strong refining margins."},
                    {"title": "India-Russia oil deal signed", "summary": "Major crude supply agreement announced."}
                ]
            }
        }
    }


@app.get("/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def serve_frontend():
    """Serve the frontend dashboard at root so opening 127.0.0.1:8000 loads the UI."""
    index_file = PROJECT_ROOT / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="frontend/index.html not found")


@app.get("/api/info", tags=["meta"])
def api_info():
    """Returns API status, available endpoints, and active provider info."""
    from news.provider import get_active_provider_info
    p_info = get_active_provider_info()
    return {
        "message": "FinSight AI Prediction & News Intelligence API is running",
        "dashboard_url": "/",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "prediction": "POST /api/v1/predict",
            "predict_with_news": "POST /api/v1/predict-with-news",
            "predict_live": "GET /api/v1/predict-live/{symbol}",
            "predict_live_yf": "GET /api/v1/predict-live-yf/{symbol}",
            "yf_validate": "GET /api/v1/yfinance/validate/{symbol}",
            "yf_historical": "GET /api/v1/yfinance/historical/{symbol}",
            "symbols": "GET /api/v1/symbols",
            "company_profile": "GET /api/v1/company-profile/{symbol}",
            "news_live": "GET /api/v1/news/live/{symbol}",
            "news_live_features": "GET /api/v1/news/live/{symbol}/features",
            "news_analyze": "POST /api/v1/news/analyze",
            "news_analyze_text": "POST /api/v1/news/analyze-text",
            "news_features": "POST /api/v1/news/features",
        },
        **p_info,
    }


@app.get("/health")
def health():
    from news.provider import get_active_provider_info
    return {
        "status": "UP",
        "news_engine": "enabled",
        **get_active_provider_info(),
    }


@app.get("/api/v1/symbols")
def get_symbols():
    """Return list of stock symbols available for prediction (based on CSVs in data/raw/)."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return {"symbols": []}
    available = {f.stem for f in raw_dir.glob("*.csv")}

    # Preferred display order
    preferred_order = ["RELIANCE", "INFY_NS", "TCS_NS", "HDFCBANK_NS"]
    symbols = [s for s in preferred_order if s in available]
    symbols += sorted([s for s in available if s not in preferred_order])

    # Annotate with profile info where available
    result = []
    for s in symbols:
        profile = get_profile(s)
        curr, curr_sym = get_currency(s)
        yf_sym = normalize_to_yf(s)
        result.append({
            "symbol": s,
            "name": profile["name"],
            "sector": profile["sector"],
            "has_profile": bool(profile["name"]),
            "currency": curr,
            "currency_symbol": curr_sym,
            "has_csv": True,
            "yf_symbol": yf_sym,
        })
    return {"symbols": result}


@app.get("/api/v1/company-profile/{symbol}")
def company_profile(symbol: str):
    """Return the semantic knowledge-base profile for a given stock symbol."""
    profile = get_profile(symbol)
    return {
        "symbol": symbol,
        "canonical_symbol": normalize_symbol(symbol),
        "name": profile["name"],
        "sector": profile["sector"],
        "description": profile["description"],
        "sector_tags": profile["sector_tags"],
        "theme_exposures": profile["themes"],
    }



@app.get("/api/v1/analyze/{symbol}")
def analyze_stock(
    symbol: str,
    session: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    """
    Fetch market data for a symbol (using Yahoo Finance with CSV fallback)
    and return multi-horizon forecasts (1W / 15D / 1M) powered by dedicated
    retrained XGBoost models.

    Open to all logged-in users — no symbol restriction. Every analysis is logged to the
    audit DB so the admin panel can track who analyzed what.
    """
    sym_upper = symbol.upper()

    # ── Quota enforcement: regular users limited to 7 analyses ───────────────
    check_analysis_quota(db=db, session=session, symbol=sym_upper)

    try:
        df = None
        data_source = "yfinance"
        company_name = sym_upper

        # 1. Try Yahoo Finance live fetch
        try:
            from ingestion.yfinance_provider import YFinanceMarketDataProvider
            provider = YFinanceMarketDataProvider()
            df = provider.get_historical_data(symbol, period="1y")
            yf_info = get_yf_ticker_info(symbol)
            company_name = yf_info.get("shortName") or yf_info.get("longName") or sym_upper
        except Exception as yf_err:
            logger.info("[Analyze] YF fetch failed for %s: %s, trying CSV fallback...", symbol, yf_err)

        # 2. Fallback to local historical CSV
        if df is None or len(df) == 0:
            price_path = resolve_price_csv(symbol)
            if price_path and price_path.exists():
                df = CSVMarketDataProvider().get_historical_data(str(price_path))
                data_source = "csv"

        if df is None or len(df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Could not load market data for '{symbol}' from Yahoo Finance or local CSVs."
            )

        from ml.predict import predict_multi_horizon
        result = predict_multi_horizon(df, symbol=symbol)
        result["company_name"] = company_name
        result["data_source"] = data_source

        # ── Log activity to audit database ────────────────────────────────────
        # horizons is a LIST of dicts: [{label, direction, target_price, ...}, ...]
        # Keys stored as '1W', '15D', '1M' for consistent frontend lookup.
        try:
            _KEY_MAP = {"Next Day": "1D", "1 Week": "1W", "15 Days": "15D", "1 Month": "1M"}
            horizons_list = result.get("horizons") or []
            summary = {
                _KEY_MAP.get(h.get("label", ""), h.get("label", f"h{i}")): {
                    "direction":        h.get("direction"),
                    "prob_up":          round(h.get("probability_up", 0), 3),
                    "prob_neutral":     round(h.get("probability_neutral", 0), 3),
                    "prob_down":        round(h.get("probability_down", 0), 3),
                    "target_price":     h.get("target_price"),
                    "price_change_pct": h.get("price_change_pct"),
                }
                for i, h in enumerate(horizons_list)
                if isinstance(h, dict)
            }
            log_activity(
                db=db,
                symbol=sym_upper,
                action="stock_analysis",
                details={
                    "company_name": company_name,
                    "data_source":  data_source,
                    "close":        result.get("close"),
                    "currency_symbol": result.get("currency_symbol", ""),
                    "horizons":     summary,
                    # news_adj is populated only by news-enhanced endpoints; leave absent here
                },
                session=session,
            )
        except Exception as log_err:
            logger.warning("[ActivityLog] Failed to log: %s", log_err)

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/predict")
def prediction(
    request: PredictionRequest,
    session: dict = Depends(require_user),
):
    price_path = resolve_file(request.csv_file)
    if not price_path or not Path(price_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Price CSV file not found: '{request.csv_file}'. Expected an existing CSV with date, open, high, low, close, volume columns."
        )

    try:
        df = CSVMarketDataProvider().get_historical_data(price_path)
        news_features = None
        if request.news_csv_file:
            news_path = resolve_file(request.news_csv_file)
            if not news_path or not Path(news_path).exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"News CSV file not found: '{request.news_csv_file}'."
                )
            from news.ingest import load_news_csv
            from news.features import aggregate_features
            news_features = aggregate_features(load_news_csv(news_path), symbol=request.symbol)
        return {"success": True, "prediction": predict(df, news_features, symbol=request.symbol)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/predict-with-news")
def predict_with_news(
    request: PredictWithNewsRequest,
    session: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Predict stock direction using a symbol name + inline news text (no CSV paths needed)."""

    # ── Quota enforcement: regular users limited to 7 analyses ───────────────
    check_analysis_quota(db=db, session=session, symbol=request.symbol)

    # Locate price CSV by symbol name
    price_path = resolve_price_csv(request.symbol)
    if not price_path:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        raise HTTPException(
            status_code=400,
            detail=f"No price CSV found for symbol '{request.symbol}'. Available: {[f.stem for f in raw_dir.glob('*.csv')]}"
        )

    try:
        from news.models import NewsArticle
        from news.features import aggregate_features, analyze_articles

        df = CSVMarketDataProvider().get_historical_data(str(price_path))

        # Build NewsArticle objects from inline items
        articles = [
            NewsArticle(
                title=item.title,
                summary=item.summary or "",
                source=item.source or "",
                url=item.url or "",
                symbol=request.symbol,
            )
            for item in request.news_items
        ]

        if articles:
            try:
                from news.alpha_vantage import save_today_news_files
                save_today_news_files(articles, [request.symbol], provider="inline_analysis")
            except Exception as err:
                logger.warning("Could not persist inline articles to today_news.md: %s", err)

        news_features = aggregate_features(articles, symbol=request.symbol) if articles else None

        # Per-article NLP analysis for the response
        article_analysis = []
        if articles:
            analyzed_df = analyze_articles(articles)
            for row in analyzed_df.to_dict(orient="records"):
                article_analysis.append({
                    k: (None if isinstance(v, float) and v != v else v)
                    for k, v in row.items()
                })

        # Company-aware semantic impact analysis
        article_dicts = [
            {"title": item.title, "summary": item.summary or ""}
            for item in request.news_items
        ]
        company_impact = analyze_company_impact(article_dicts, request.symbol)

        # Merge company semantic adjustment into news_features so the predictor uses it
        merged_features = dict(news_features or {})

        if company_impact.has_company_specific_impact and merged_features:
            base_sentiment = merged_features.get("news_sentiment_1d", 0.0)
            adj = company_impact.company_sentiment_adjustment
            # Weighted blend: 50% NLP sentiment + 50% company semantic signal
            # (increased from 40% to give more weight to company-specific context)
            merged_features["news_sentiment_1d"] = round(0.50 * base_sentiment + 0.50 * adj, 4)
            merged_features["news_impact_1d"] = round(
                merged_features.get("news_impact_1d", 0.0) + adj * 0.4, 4
            )

        # Always propagate Marketaux avg sentiment if present -- the predict function
        # prioritises this over the NLP lexicon signal, so ensure it survives the merge.
        # (It is already in merged_features since we copied from news_features above.)

        result = predict(df, merged_features if merged_features else news_features, symbol=request.symbol)

        # Log activity - consolidates into existing session row if already analyzed
        try:
            inline_news_adj = {
                "direction":        result.get("label"),
                "target_price":     result.get("target_price"),
                "price_change_pct": result.get("price_change_pct"),
                "news_source":      "inline_news",
            } if result.get("target_price") else None
            log_activity(
                db=db,
                symbol=request.symbol.upper(),
                action="stock_analysis",
                details={
                    "source":          "predict-with-news",
                    "news_source":     "inline_news",
                    "close":           result.get("close"),
                    "currency_symbol": result.get("currency_symbol", ""),
                    "horizons":        {},
                    "news_adj":        inline_news_adj,
                },
                session=session,
            )
        except Exception as log_err:
            logger.warning("[ActivityLog] predict-with-news log failed: %s", log_err)

        return {
            "success": True,
            "prediction": result,
            "news_features": merged_features or {},
            "article_analysis": article_analysis,
            "company_impact": company_impact.to_dict(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Live Prediction Endpoint (Alpha Vantage / Unified Provider) ──────────────

@app.get("/api/v1/predict-live/{symbol}")
def predict_live(
    symbol: str,
    days_back: int = 2,
    use_cache: bool = True,
    session: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Fully automatic: fetch live news from Alpha Vantage, analyze it, and predict stock direction.

    No CSV paths or manual news input needed. Just pass the stock symbol.
    Uses Alpha Vantage News & Sentiment API as primary provider, with Marketaux fallback.

    Args:
        symbol:    Stock ticker symbol (e.g. MSFT, NVDA, RELIANCE, INFY).
        days_back: Days of news history to fetch (default 2).
        use_cache: Whether to use cached articles within the 30-minute TTL.
    """

    # ── Quota enforcement: regular users limited to 7 analyses ───────────────
    check_analysis_quota(db=db, session=session, symbol=symbol)

    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "") or os.getenv("ALPHAVANTAGE_API_KEY", "")
    mx_key = os.getenv("MARKETAUX_API_KEY", "")
    if not av_key and not mx_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "No News API key configured. "
                "Add ALPHA_VANTAGE_API_KEY=<your_key> to your .env file and restart the server."
            ),
        )

    # Locate price CSV by symbol name — fall back to Yahoo Finance if missing
    price_path = resolve_price_csv(symbol)
    use_yf_fallback = False
    if not price_path:
        logger.info("predict_live: no local CSV for '%s'; falling back to Yahoo Finance", symbol)
        use_yf_fallback = True

    try:
        from news.provider import fetch_live_news
        from news.features import aggregate_features, analyze_articles
        from news.semantic import analyze_company_impact
        from news.api import _score_article_relevance

        if use_yf_fallback:
            try:
                df = YFinanceMarketDataProvider().get_historical_data(symbol, period="2y")
            except Exception as yf_exc:
                raw_dir = PROJECT_ROOT / "data" / "raw"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No local CSV found for '{symbol}' and Yahoo Finance fetch also failed: {yf_exc}. "
                        f"Local CSVs available: {[f.stem for f in raw_dir.glob('*.csv')]}"
                    ),
                )
        else:
            df = CSVMarketDataProvider().get_historical_data(str(price_path))

        # Fetch live news from Alpha Vantage / active provider with graceful fallback
        try:
            articles, provider_name = fetch_live_news(
                symbols=[symbol],
                days_back=max(1, min(days_back, 7)),
                use_cache=use_cache,
            )
        except Exception as exc:
            logger.warning("fetch_live_news failed for %s: %s; continuing with technical indicators", symbol, exc)
            articles, provider_name = [], "technical_fallback"

        # ── Score relevance for every article ────────────────────────────────
        # Only RELEVANT articles (score >= 0.35) influence the prediction.
        # General market news is shown in the UI but does NOT move the model.
        relevant_articles = []
        general_articles  = []
        article_analysis  = []

        if articles:
            analyzed_df = analyze_articles(articles)
            analyzed_records = [
                {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
                for row in analyzed_df.to_dict(orient="records")
            ]
            for rec in analyzed_records:
                scored = _score_article_relevance(rec, symbol)
                article_analysis.append(scored)
                if scored["is_relevant"]:
                    relevant_articles.append(scored)
                else:
                    general_articles.append(scored)

        # Build NewsArticle objects only for relevant articles so the predictor
        # uses a clean, company-focused signal.
        from news.models import NewsArticle
        relevant_news_objs = [
            next((a for a in articles if a.title == r["title"]), None)
            for r in relevant_articles
        ]
        relevant_news_objs = [a for a in relevant_news_objs if a is not None]

        # Aggregate features from relevant-only articles
        if relevant_news_objs:
            news_features = aggregate_features(relevant_news_objs, symbol=symbol)
        elif articles:
            # No relevant articles found — fall back to all articles but heavily discount
            news_features = aggregate_features(articles, symbol=symbol)
            # Discount the sentiment since these are not company-specific
            news_features["news_sentiment_1d"] = round(news_features.get("news_sentiment_1d", 0.0) * 0.3, 4)
            news_features["news_impact_1d"]    = round(news_features.get("news_impact_1d", 0.0) * 0.3, 4)
        else:
            news_features = None

        # ── Company-aware semantic impact (relevant articles only) ────────────
        relevant_dicts = [{"title": r["title"], "summary": r.get("summary") or ""} for r in relevant_articles]
        all_dicts      = [{"title": r["title"], "summary": r.get("summary") or ""} for r in article_analysis]
        company_impact = analyze_company_impact(relevant_dicts or all_dicts, symbol)

        # ── Merge semantic adjustment into news_features ──────────────────────
        merged_features = dict(news_features or {})

        if company_impact.has_company_specific_impact and merged_features:
            base_sentiment = merged_features.get("news_sentiment_1d", 0.0)
            adj = company_impact.company_sentiment_adjustment
            merged_features["news_sentiment_1d"] = round(0.50 * base_sentiment + 0.50 * adj, 4)
            merged_features["news_impact_1d"]    = round(
                merged_features.get("news_impact_1d", 0.0) + adj * 0.4, 4
            )

        result = predict(df, merged_features if merged_features else news_features, symbol=symbol)

        # Log activity - predict-live
        try:
            news_adj = {
                "direction":        result.get("label"),
                "target_price":     result.get("target_price"),
                "price_change_pct": result.get("price_change_pct"),
                "news_source":      provider_name,
            } if result.get("target_price") else None
            log_activity(
                db=db,
                symbol=symbol.upper(),
                action="stock_analysis",
                details={
                    "source":          "predict-live",
                    "news_source":     provider_name,
                    "close":           result.get("close"),
                    "currency_symbol": result.get("currency_symbol", ""),
                    "horizons":        {},
                    "news_adj":        news_adj,
                },
                session=session,
            )
        except Exception as log_err:
            logger.warning("[ActivityLog] predict-live log failed: %s", log_err)

        return {
            "success": True,
            "symbol": symbol,
            "news_source": provider_name,
            "article_count":   len(articles),
            "relevant_count":  len(relevant_articles),
            "general_count":   len(general_articles),
            "days_back": days_back,
            "prediction": result,
            "news_features": merged_features or {},
            # Full analysis list (all articles with relevance metadata)
            "article_analysis":   article_analysis,
            # Segregated for the frontend panels
            "relevant_articles":  relevant_articles,
            "general_articles":   general_articles,
            "company_impact": company_impact.to_dict(),
        }
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except RuntimeError as exc:
        # Handles rate-limit and other Marketaux errors
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Yahoo Finance Endpoints ──────────────────────────────────────────────────

@app.get("/api/v1/stocks/search")
def search_stocks(
    q: str = Query(default="", description="Search query by name or ticker"),
    session: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Search stocks by company name, ticker symbol, or sector.
    Returns matches from both the approved benchmark database and Yahoo Finance global market.
    """
    q_clean = (q or "").strip()
    results = []
    seen_symbols = set()

    # 1. Search in local approved benchmark datasets
    try:
        allowed = db.query(AllowedSymbol).all()
    except Exception:
        allowed = []

    allowed_set = {a.symbol.upper() for a in allowed}

    for s in allowed:
        if not q_clean:
            match = True
        else:
            match = (
                q_clean.lower() in s.symbol.lower()
                or q_clean.lower() in (s.display_name or "").lower()
                or q_clean.lower() in (s.sector or "").lower()
                or q_clean.lower() in (s.exchange or "").lower()
            )
        if match:
            results.append({
                "symbol": s.symbol,
                "name": s.display_name or s.symbol,
                "sector": s.sector or "",
                "exchange": s.exchange or "",
                "is_benchmark": True,
                "is_allowed": True,
                "source": "benchmark"
            })
            seen_symbols.add(s.symbol.upper())

    # 2. If query provided, search Yahoo Finance for broader global results
    if q_clean and len(q_clean) >= 2:
        try:
            import yfinance as yf
            search_obj = yf.Search(q_clean, max_results=6)
            quotes = getattr(search_obj, "quotes", []) or []

            for item in quotes:
                quote_type = item.get("quoteType", "")
                if quote_type not in ("EQUITY", "ETF"):
                    continue
                sym = (item.get("symbol") or "").upper()
                if not sym or sym in seen_symbols:
                    continue

                name = item.get("shortname") or item.get("longname") or sym
                exch = item.get("exchDisp") or item.get("exchange") or ""
                sector = item.get("sectorDisp") or item.get("sector") or item.get("industry") or ""
                is_allowed = (sym in allowed_set)

                results.append({
                    "symbol": sym,
                    "name": name,
                    "sector": sector,
                    "exchange": exch,
                    "is_benchmark": is_allowed,
                    "is_allowed": is_allowed,
                    "source": "yahoo_finance"
                })
                seen_symbols.add(sym)
        except Exception as exc:
            logger.warning("Yahoo Finance search failed for '%s': %s", q_clean, exc)

    return {"query": q_clean, "count": len(results), "results": results}


@app.get("/api/v1/yfinance/validate/{symbol}")
def yf_validate(
    symbol: str,
    session: dict = Depends(require_user),
):
    """Validate whether a ticker symbol exists on Yahoo Finance.

    Returns metadata (company name, sector, currency, exchange) when valid.
    Used by the frontend to confirm custom ticker inputs before predicting.
    """
    try:
        is_valid, canonical, meta = validate_yf_ticker(symbol)
        return {
            "symbol": symbol,
            "canonical_yf_symbol": canonical,
            "is_valid": is_valid,
            "name": meta.get("name", ""),
            "sector": meta.get("sector", ""),
            "industry": meta.get("industry", ""),
            "currency": meta.get("currency", ""),
            "exchange": meta.get("exchange", ""),
            "market_cap": meta.get("market_cap"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/yfinance/historical/{symbol}")
def yf_historical(
    symbol: str,
    period: str = Query(default="1y", description="Data period (1mo, 3mo, 6mo, 1y, 2y, 5y)"),
    session: dict = Depends(require_user),
):
    """Fetch historical OHLCV data from Yahoo Finance for any ticker.

    Useful for the frontend to display price charts without a local CSV.
    Returns the last N rows sorted newest-first for display.
    """
    try:
        provider = YFinanceMarketDataProvider()
        df = provider.get_historical_data(symbol, period=period)
        # Return newest first for easier frontend consumption
        df_out = df.sort_values("date", ascending=False).head(500)
        records = []
        for _, row in df_out.iterrows():
            records.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": int(row["volume"]),
            })
        return {
            "success": True,
            "symbol": symbol,
            "canonical_yf_symbol": normalize_to_yf(symbol),
            "period": period,
            "row_count": len(records),
            "data": records,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/predict-live-yf/{symbol}")
def predict_live_yf(
    symbol: str,
    days_back: int = Query(default=2, ge=1, le=7, description="Days of news history to fetch"),
    period: str = Query(default="2y", description="Historical price data period for Yahoo Finance"),
    use_cache: bool = Query(default=True, description="Whether to use cached news results"),
    session: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Fully live prediction using Yahoo Finance for price data + Alpha Vantage for news.

    Unlike /api/v1/predict-live/{symbol}, this endpoint NEVER requires a local CSV file.
    Any valid Yahoo Finance ticker works: AAPL, TSLA, RELIANCE.NS, TCS.NS, etc.

    Args:
        symbol:    Yahoo Finance ticker (e.g. MSFT, NVDA, RELIANCE.NS, TCS.NS).
        days_back: Days of news history to fetch (default 2).
        period:    How much price history to fetch from Yahoo Finance (default 2y).
        use_cache: Whether to use cached news articles (30-min TTL).
    """

    # ── Quota enforcement: regular users limited to 7 analyses ───────────────
    check_analysis_quota(db=db, session=session, symbol=symbol)

    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "") or os.getenv("ALPHAVANTAGE_API_KEY", "")
    mx_key = os.getenv("MARKETAUX_API_KEY", "")
    if not av_key and not mx_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "No News API key configured. "
                "Add ALPHA_VANTAGE_API_KEY=<your_key> to your .env file and restart the server."
            ),
        )

    try:
        from news.provider import fetch_live_news
        from news.features import aggregate_features, analyze_articles
        from news.semantic import analyze_company_impact
        from news.api import _score_article_relevance
        from news.models import NewsArticle

        # ── Step 1: Fetch price data from Yahoo Finance ───────────────────────────
        try:
            df = YFinanceMarketDataProvider().get_historical_data(symbol, period=period)
        except Exception as yf_exc:
            raise HTTPException(
                status_code=400,
                detail=f"Yahoo Finance data fetch failed for '{symbol}': {yf_exc}"
            )

        # Get company metadata from Yahoo Finance for display
        yf_info = get_yf_ticker_info(symbol)
        company_name = yf_info.get("shortName") or yf_info.get("longName") or symbol
        yf_currency = yf_info.get("currency", "")
        yf_currency_sym = "$" if yf_currency == "USD" else ("₹" if yf_currency == "INR" else yf_currency)

        # ── Step 2: Fetch live news scoped to this ticker ─────────────────────────
        try:
            articles, provider_name = fetch_live_news(
                symbols=[symbol],
                days_back=max(1, min(days_back, 7)),
                use_cache=use_cache,
            )
        except Exception as exc:
            logger.warning("fetch_live_news failed for %s: %s; continuing without news", symbol, exc)
            articles, provider_name = [], "technical_fallback"

        # ── Step 3: Score article relevance ──────────────────────────────────────
        relevant_articles = []
        general_articles  = []
        article_analysis  = []

        if articles:
            analyzed_df = analyze_articles(articles)
            analyzed_records = [
                {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
                for row in analyzed_df.to_dict(orient="records")
            ]
            for rec in analyzed_records:
                scored = _score_article_relevance(rec, symbol)
                article_analysis.append(scored)
                if scored["is_relevant"]:
                    relevant_articles.append(scored)
                else:
                    general_articles.append(scored)

        relevant_news_objs = [
            next((a for a in articles if a.title == r["title"]), None)
            for r in relevant_articles
        ]
        relevant_news_objs = [a for a in relevant_news_objs if a is not None]

        # ── Step 4: Aggregate news features ──────────────────────────────────────
        if relevant_news_objs:
            news_features = aggregate_features(relevant_news_objs, symbol=symbol)
        elif articles:
            news_features = aggregate_features(articles, symbol=symbol)
            news_features["news_sentiment_1d"] = round(news_features.get("news_sentiment_1d", 0.0) * 0.3, 4)
            news_features["news_impact_1d"]    = round(news_features.get("news_impact_1d", 0.0) * 0.3, 4)
        else:
            news_features = None

        # ── Step 5: Company-aware semantic impact ─────────────────────────────────
        relevant_dicts = [{"title": r["title"], "summary": r.get("summary") or ""} for r in relevant_articles]
        all_dicts      = [{"title": r["title"], "summary": r.get("summary") or ""} for r in article_analysis]
        company_impact = analyze_company_impact(relevant_dicts or all_dicts, symbol)

        # ── Step 6: Merge semantic signal into features ───────────────────────────
        merged_features = dict(news_features or {})
        if company_impact.has_company_specific_impact and merged_features:
            base_sentiment = merged_features.get("news_sentiment_1d", 0.0)
            adj = company_impact.company_sentiment_adjustment
            merged_features["news_sentiment_1d"] = round(0.50 * base_sentiment + 0.50 * adj, 4)
            merged_features["news_impact_1d"]    = round(
                merged_features.get("news_impact_1d", 0.0) + adj * 0.4, 4
            )

        # ── Step 7: Run ML prediction ─────────────────────────────────────────────
        result = predict(df, merged_features if merged_features else news_features, symbol=symbol)

        # Override currency in result if yfinance provides it
        if yf_currency and isinstance(result, dict):
            result["currency"] = yf_currency
            result["currency_symbol"] = yf_currency_sym

        # Log activity - predict-live-yf
        # NOTE: predict-live-yf uses predict() (single-horizon, news-adjusted).
        # There is no multi-horizon breakdown here; store result as news_adj.
        try:
            yf_news_adj = {
                "direction":        result.get("label"),
                "target_price":     result.get("target_price"),
                "price_change_pct": result.get("price_change_pct"),
                "news_source":      provider_name,
                "company":          company_name,
            } if result.get("target_price") else None
            log_activity(
                db=db,
                symbol=symbol.upper(),
                action="stock_analysis",
                details={
                    "source":          "predict-live-yf",
                    "company":         company_name,
                    "close":           result.get("close"),
                    "currency_symbol": result.get("currency_symbol", yf_currency_sym),
                    "horizons":        {},
                    "news_adj":        yf_news_adj,
                },
                session=session,
            )
        except Exception as log_err:
            logger.warning("[ActivityLog] predict-live-yf log failed: %s", log_err)


        return {
            "success": True,
            "symbol": symbol,
            "canonical_yf_symbol": normalize_to_yf(symbol),
            "company_name": company_name,
            "data_source": "yahoo_finance",
            "news_source": provider_name,
            "price_period": period,
            "price_rows": len(df),
            "article_count":  len(articles),
            "relevant_count": len(relevant_articles),
            "general_count":  len(general_articles),
            "days_back": days_back,
            "prediction": result,
            "news_features": merged_features or {},
            "article_analysis":  article_analysis,
            "relevant_articles": relevant_articles,
            "general_articles":  general_articles,
            "company_impact": company_impact.to_dict(),
        }
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

