"""Optional PostgreSQL persistence for article/NLP outputs."""
from sqlalchemy import create_engine, text

SCHEMA = """
CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    content TEXT,
    source VARCHAR(255),
    url TEXT,
    published_at TIMESTAMPTZ,
    symbol VARCHAR(50),
    sector VARCHAR(100),
    language VARCHAR(10) DEFAULT 'en',
    content_hash CHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_news_articles_symbol_time ON news_articles(symbol, published_at);
CREATE INDEX IF NOT EXISTS idx_news_articles_time ON news_articles(published_at);

CREATE TABLE IF NOT EXISTS news_analysis (
    id BIGSERIAL PRIMARY KEY,
    news_id BIGINT NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    sentiment NUMERIC(10,8), confidence NUMERIC(10,8), importance NUMERIC(10,8),
    event_type VARCHAR(50), impact NUMERIC(10,8), method VARCHAR(30),
    positive_hits INTEGER DEFAULT 0, negative_hits INTEGER DEFAULT 0,
    keywords_json JSONB, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(news_id)
);

CREATE TABLE IF NOT EXISTS news_feature_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50), feature_time TIMESTAMPTZ NOT NULL,
    features_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_news_features_symbol_time ON news_feature_snapshots(symbol, feature_time);
"""

def init_news_schema(database_url: str):
    engine = create_engine(database_url)
    with engine.begin() as conn:
        for statement in SCHEMA.split(";"):
            if statement.strip():
                conn.execute(text(statement))
