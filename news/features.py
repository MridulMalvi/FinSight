"""Turn article-level NLP outputs into time-decayed stock/market features.

When articles carry a Marketaux API sentiment score (marketaux_sentiment),
that score is blended with the lexicon NLP result:
    final_sentiment = 0.60 × marketaux_sentiment + 0.40 × nlp_sentiment

This hybrid signal is more accurate because Marketaux's model is domain-trained
specifically on financial news.
"""
from datetime import datetime, timezone
import math
import pandas as pd
from news.models import NewsArticle
from news.nlp import NewsAnalyzer


def _hours_old(published_at, now):
    if not published_at:
        return 24.0
    dt = published_at if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 3600.0)


def analyze_articles(articles, analyzer=None):
    analyzer = analyzer or NewsAnalyzer()
    rows = []
    for a in articles:
        x = analyzer.analyze(a.text, source=a.source)
        row = {**a.to_dict(), **x.to_dict()}
        # Annotate with blended sentiment when Marketaux score is available
        if a.marketaux_sentiment is not None:
            row["blended_sentiment"] = round(
                0.60 * a.marketaux_sentiment + 0.40 * x.sentiment, 4
            )
        else:
            row["blended_sentiment"] = round(x.sentiment, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _blended_sentiment(article: NewsArticle, nlp_sentiment: float) -> float:
    """Return hybrid sentiment: Marketaux (60%) + NLP (40%) when API score is available."""
    if article.marketaux_sentiment is not None:
        return 0.60 * article.marketaux_sentiment + 0.40 * nlp_sentiment
    return nlp_sentiment


def aggregate_features(articles, symbol=None, as_of=None, half_life_hours=24.0):
    analyzer = NewsAnalyzer()
    now = as_of or datetime.now(timezone.utc)
    # Include articles that either:
    #  a) have no symbol tag (untagged general news — include as context)
    #  b) have a symbol that matches the target
    # Only exclude articles explicitly tagged to a DIFFERENT company.
    def _article_matches(a) -> bool:
        if not symbol:
            return True
        if not a.symbol and not getattr(a, "symbols", None):  # untagged → always include
            return True

        q_clean = symbol.upper().split(".")[0].replace("_HISTORICAL", "").replace("_HIST", "").replace("_NS", "").replace("_BO", "")
        allowed_q = {q_clean, symbol.upper()}
        if "RELIANCE" in q_clean:
            allowed_q.add("RELIANCE")
        if "INFOSYS" in q_clean or q_clean == "INFY":
            allowed_q.update(["INFY", "INFOSYS"])
        if "MSFT" in q_clean:
            allowed_q.update(["MSFT", "MICROSOFT"])
        if "AAPL" in q_clean:
            allowed_q.update(["AAPL", "APPLE"])
        if q_clean in ("GOOGL", "GOOG"):
            allowed_q.update(["GOOGL", "GOOG", "GOOGLE", "ALPHABET"])
        if "NVDA" in q_clean:
            allowed_q.update(["NVDA", "NVIDIA"])
        if "TCS" in q_clean:
            allowed_q.update(["TCS", "TATA"])
        if "TATAMOTORS" in q_clean:
            allowed_q.update(["TATAMOTORS", "TATA", "JLR"])
        if "HDFC" in q_clean:
            allowed_q.update(["HDFCBANK", "HDFC"])
        if "ICICI" in q_clean:
            allowed_q.update(["ICICIBANK", "ICICI"])
        if "TSLA" in q_clean:
            allowed_q.update(["TSLA", "TESLA"])

        # Target 10 companies entity and brand name matching
        if "ASHOK" in q_clean or "LEYLAND" in q_clean:
            allowed_q.update(["ASHOKLEY", "ASHOK LEYLAND", "LEYLAND", "SWITCH MOBILITY"])
        if "BEL" in q_clean or "BHARAT_ELEC" in q_clean:
            allowed_q.update(["BEL", "BHARAT ELECTRONICS", "BHARAT ELECTRONIC"])
        if "GOLD" in q_clean:
            allowed_q.update(["GOLDBEES", "GOLD BEES", "GOLD ETF", "GOLD PRICE", "BULLION", "SPOT GOLD"])
        if "HUDCO" in q_clean:
            allowed_q.update(["HUDCO", "HOUSING AND URBAN DEVELOPMENT", "HOUSING & URBAN DEVELOPMENT"])
        if "IDBI" in q_clean:
            allowed_q.update(["IDBI", "IDBIBANK", "IDBI BANK"])
        if "IOC" in q_clean or "INDIANOIL" in q_clean or "INDIAN_OIL" in q_clean:
            allowed_q.update(["IOC", "IOCL", "INDIANOIL", "INDIAN OIL", "INDIAN OIL CORPORATION"])
        if "IRFC" in q_clean:
            allowed_q.update(["IRFC", "INDIAN RAILWAY FINANCE", "RAILWAY FINANCE"])
        if "MID150" in q_clean or "MIDCAP" in q_clean:
            allowed_q.update(["MID150BEES", "MID150", "NIFTY MIDCAP 150", "MIDCAP 150", "MIDCAP ETF", "MIDCAP"])
        if "POWERGRID" in q_clean or "PGCIL" in q_clean or "POWER_GRID" in q_clean:
            allowed_q.update(["POWERGRID", "POWER GRID", "PGCIL", "POWER GRID CORPORATION"])
        if "SUZLON" in q_clean:
            allowed_q.update(["SUZLON", "SUZLON ENERGY"])

        # Check primary symbol
        a_sym = (a.symbol or "").upper().split(".")[0].replace("_NS", "").replace("_BO", "")
        if a_sym in allowed_q:
            return True
        # Check all tagged entity symbols
        art_syms = [s.upper().split(".")[0].replace("_NS", "").replace("_BO", "") for s in (getattr(a, "symbols", None) or [])]
        if any(s in allowed_q for s in art_syms):
            return True
        # Check company name or ticker in text
        text_lc = (a.text or "").lower()
        if any(len(s) > 2 and s.lower() in text_lc for s in allowed_q):
            return True
        return False


    selected = [a for a in articles if _article_matches(a)]
    if not selected:
        return _empty_features()
    analyzed = []
    for a in selected:
        x = analyzer.analyze(a.text, source=a.source)
        age = _hours_old(a.published_at, now)
        # Decay: articles older than 48h get very low weight; inline/live news (age≈0) gets full weight
        decay = max(0.05, 0.5 ** (age / max(half_life_hours, 0.1)))
        # Apply hybrid blending when Marketaux score is present
        effective_sentiment = _blended_sentiment(a, x.sentiment)
        analyzed.append((a, x, decay, effective_sentiment))

    weights = [d for _, _, d, _ in analyzed]
    total_w = sum(weights) or 1.0

    # Use impact-weighted sentiment: high-magnitude articles dominate
    impact_weights = [abs(x.impact) * d + 0.01 for _, x, d, _ in analyzed]
    total_iw = sum(impact_weights) or 1.0
    sentiment = sum(eff_s * impact_weights[i] for i, (_, _, _, eff_s) in enumerate(analyzed)) / total_iw

    importance = sum(x.importance * d for _, x, d, _ in analyzed) / total_w
    impact     = sum(x.impact * d for _, x, d, _ in analyzed) / total_w
    pos = sum(1 for _, _, _, eff_s in analyzed if eff_s > 0.05)
    neg = sum(1 for _, _, _, eff_s in analyzed if eff_s < -0.05)
    event_counts: dict = {}
    for _, x, _, _ in analyzed:
        event_counts[x.event_type] = event_counts.get(x.event_type, 0) + 1

    # Average Marketaux-provided sentiment (articles that have it)
    mx_scores = [a.marketaux_sentiment for a, _, _, _ in analyzed if a.marketaux_sentiment is not None]
    marketaux_sentiment_avg = sum(mx_scores) / len(mx_scores) if mx_scores else None

    return {
        "news_sentiment_1d": float(sentiment),
        "news_impact_1d": float(impact),
        "news_importance_1d": float(importance),
        "news_volume_1d": int(len(analyzed)),
        "positive_news_ratio_1d": float(pos / len(analyzed)),
        "negative_news_ratio_1d": float(neg / len(analyzed)),
        "earnings_news_1d": int(event_counts.get("EARNINGS", 0)),
        "regulatory_news_1d": int(event_counts.get("REGULATORY", 0)),
        "legal_news_1d": int(event_counts.get("LEGAL", 0)),
        "contract_news_1d": int(event_counts.get("CONTRACT_ORDER", 0)),
        "management_news_1d": int(event_counts.get("MANAGEMENT", 0)),
        "macro_news_1d": int(event_counts.get("MACRO", 0)),
        # Marketaux-specific features (None when no Marketaux articles present)
        "marketaux_sentiment_avg": marketaux_sentiment_avg,
        "marketaux_articles_count": int(len(mx_scores)),
    }


def _empty_features():
    return {
        "news_sentiment_1d": 0.0, "news_impact_1d": 0.0, "news_importance_1d": 0.0,
        "news_volume_1d": 0, "positive_news_ratio_1d": 0.0, "negative_news_ratio_1d": 0.0,
        "earnings_news_1d": 0, "regulatory_news_1d": 0, "legal_news_1d": 0,
        "contract_news_1d": 0, "management_news_1d": 0, "macro_news_1d": 0,
        "marketaux_sentiment_avg": None, "marketaux_articles_count": 0,
    }
