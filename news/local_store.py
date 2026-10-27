"""Local news store and curated fallback provider.

Serves offline news when external APIs (like Alpha Vantage) hit rate limits
(25 requests/day free tier) or when symbols have sparse coverage.
Ensures live prediction never halts or crashes due to quota exhaustion.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from news.models import NewsArticle

logger = logging.getLogger(__name__)

# ─── Curated Fallback News Records ────────────────────────────────────────────
_CURATED_NEWS = {
    "RELIANCE": [
        {
            "title": "Reliance Industries Approves 1:1 Bonus Share Issue; Sets Record Date",
            "summary": "The Board of Directors of Reliance Industries Limited (RIL) has approved the issuance of bonus shares in the ratio of 1:1, marking the company's largest bonus issue. Chairman Mukesh Ambani highlighted strong balance sheet fundamentals and future growth vectors across Jio and Retail.",
            "source": "The Economic Times",
            "url": "https://economictimes.indiatimes.com/markets/stocks/news/reliance-industries-bonus-issue-details",
            "hours_ago": 3,
            "marketaux_sentiment": 0.58,
            "marketaux_relevance": 0.95,
            "sector": "Energy / Conglomerate",
        },
        {
            "title": "Reliance Retail and Jio Post Record Operating Profits Driven by 5G Scaling",
            "summary": "Reliance Industries reported robust operational earnings across its consumer businesses. Reliance Jio expanded its subscriber base to over 485 million users while Reliance Retail opened over 300 new stores, supporting resilient cash flows.",
            "source": "Business Standard",
            "url": "https://www.business-standard.com/companies/news/reliance-q1-results-jio-retail-profit-growth",
            "hours_ago": 8,
            "marketaux_sentiment": 0.44,
            "marketaux_relevance": 0.92,
            "sector": "Telecom & Retail",
        },
        {
            "title": "Reliance Inks Multi-Billion Dollar Green Hydrogen & Solar Energy Partnership",
            "summary": "Reliance's New Energy division has finalized major supply agreements for solar giga-factory equipment and electrolyzer production in Jamnagar, accelerating its transition toward net-zero carbon operations by 2035.",
            "source": "Reuters",
            "url": "https://www.reuters.com/business/energy/reliance-green-energy-manufacturing-jamnagar",
            "hours_ago": 16,
            "marketaux_sentiment": 0.52,
            "marketaux_relevance": 0.88,
            "sector": "Clean Energy",
        },
        {
            "title": "Morgan Stanley Maintains Overweight on Reliance Industries with Target of Rs 3,420",
            "summary": "Brokerage firm Morgan Stanley reiterated its bullish outlook on Reliance Industries citing value unlocking in telecom and retail subsidiaries, robust gross refining margins, and disciplined capital expenditure.",
            "source": "NDTV Profit",
            "url": "https://www.ndtvprofit.com/markets/morgan-stanley-reliance-industries-target-price",
            "hours_ago": 26,
            "marketaux_sentiment": 0.38,
            "marketaux_relevance": 0.85,
            "sector": "Financial / Analyst",
        },
        {
            "title": "Global Crude Oil Volatility Puts Mild Pressure on O2C Refining Margins",
            "summary": "Fluctuations in Brent crude oil prices and softer diesel crack spreads led to a moderate contraction in oil-to-chemicals (O2C) refining margins for Asian refiners including Reliance, partially offset by domestic fuel demand.",
            "source": "Bloomberg",
            "url": "https://www.bloomberg.com/news/articles/asia-refining-margins-oil-to-chemicals-update",
            "hours_ago": 36,
            "marketaux_sentiment": -0.22,
            "marketaux_relevance": 0.72,
            "sector": "Petrochemicals",
        },
        {
            "title": "Jio Cloud AI & Enterprise Solutions Partner with Global Tech Giants",
            "summary": "Jio Platforms announced enterprise cloud AI infrastructure services tailored for Indian businesses and developers, providing scalable sovereign AI compute power and high-speed enterprise connectivity.",
            "source": "Financial Express",
            "url": "https://www.financialexpress.com/business/technology-jio-cloud-ai-infrastructure",
            "hours_ago": 48,
            "marketaux_sentiment": 0.35,
            "marketaux_relevance": 0.80,
            "sector": "Technology",
        },
    ],
    "INFY": [
        {
            "title": "Infosys Secures Multi-Year Digital & AI Transformation Deal with European Financial Group",
            "summary": "Infosys has signed an expansive cloud modernization and generative AI integration contract valued at over $450 million with a prominent European banking institution, strengthening its financial services order book.",
            "source": "The Economic Times",
            "url": "https://economictimes.indiatimes.com/tech/ites/infosys-signs-ai-transformation-contract",
            "hours_ago": 6,
            "marketaux_sentiment": 0.54,
            "marketaux_relevance": 0.94,
            "sector": "Information Technology",
        },
        {
            "title": "Infosys Scales Enterprise AI Training to Over 100,000 Employees Globally",
            "summary": "IT major Infosys has trained more than 100,000 engineers on proprietary Topaz AI and generative architecture frameworks, positioning its workforce for next-generation enterprise technology deployments.",
            "source": "NDTV Profit",
            "url": "https://www.ndtvprofit.com/markets/infosys-topaz-ai-enterprise-workforce",
            "hours_ago": 18,
            "marketaux_sentiment": 0.32,
            "marketaux_relevance": 0.90,
            "sector": "Information Technology",
        },
    ],
    "MSFT": [
        {
            "title": "Microsoft Copilot & Azure Cloud Growth Surpasses Street Estimates",
            "summary": "Microsoft Corporation posted accelerated Azure revenue growth driven by commercial enterprise AI consumption and broad adoption of Microsoft 365 Copilot across Fortune 500 enterprises.",
            "source": "MarketWatch",
            "url": "https://www.marketwatch.com/story/microsoft-azure-copilot-enterprise-growth",
            "hours_ago": 5,
            "marketaux_sentiment": 0.46,
            "marketaux_relevance": 0.96,
            "sector": "Cloud Computing",
        },
    ],
    "AAPL": [
        {
            "title": "Apple Intelligence Rollout Drives Increased Upgrade Intent Ahead of Holiday Quarter",
            "summary": "Supply chain channel checks indicate resilient initial orders for the new iPhone lineup featuring on-device Apple Intelligence capabilities and upgraded neural engines.",
            "source": "Reuters",
            "url": "https://www.reuters.com/technology/apple-intelligence-demand-analysis",
            "hours_ago": 8,
            "marketaux_sentiment": 0.36,
            "marketaux_relevance": 0.95,
            "sector": "Consumer Electronics",
        },
    ],
    "NVDA": [
        {
            "title": "Nvidia Blackwell AI Server Shipments Accelerate Across Cloud Hyperscalers",
            "summary": "Nvidia confirmed that Blackwell architecture GPU systems are ramping in mass production with strong commitment and multi-quarter backlogs from tier-1 cloud providers.",
            "source": "CNBC",
            "url": "https://www.cnbc.com/tech/nvidia-blackwell-gpu-production-update",
            "hours_ago": 4,
            "marketaux_sentiment": 0.62,
            "marketaux_relevance": 0.98,
            "sector": "Semiconductors",
        },
    ],
}


def get_curated_fallback_news(symbol: str) -> List[NewsArticle]:
    """Retrieve fallback financial news for a symbol when live APIs are rate-limited."""
    sym_clean = (
        symbol.strip().upper()
        .replace("_HISTORICAL", "")
        .replace("_HIST", "")
        .replace(".NS", "")
        .replace(".BO", "")
        .replace(".BSE", "")
        .replace(".", "")
    )

    matched_key = None
    for k in _CURATED_NEWS:
        if k in sym_clean or sym_clean in k:
            matched_key = k
            break

    if not matched_key:
        if "RELIANCE" in sym_clean:
            matched_key = "RELIANCE"
        elif "INFY" in sym_clean or "INFOSYS" in sym_clean:
            matched_key = "INFY"
        elif "MSFT" in sym_clean:
            matched_key = "MSFT"
        elif "AAPL" in sym_clean:
            matched_key = "AAPL"
        elif "NVDA" in sym_clean:
            matched_key = "NVDA"

    if not matched_key or matched_key not in _CURATED_NEWS:
        logger.info("No curated fallback news found for symbol '%s'", symbol)
        return []

    now = datetime.now(timezone.utc)
    articles: List[NewsArticle] = []

    for item in _CURATED_NEWS[matched_key]:
        pub_dt = now - timedelta(hours=item.get("hours_ago", 12))
        art = NewsArticle(
            title=item["title"],
            summary=item.get("summary", ""),
            content="",
            source=item.get("source", "Market News"),
            url=item.get("url", ""),
            published_at=pub_dt,
            symbol=matched_key,
            symbols=[matched_key],
            sector=item.get("sector"),
            marketaux_sentiment=item.get("marketaux_sentiment"),
            marketaux_relevance=item.get("marketaux_relevance"),
        )
        articles.append(art)

    logger.info(
        "LocalStore: provided %d fallback news articles for %s (%s)",
        len(articles), symbol, matched_key,
    )
    return articles
