-- =============================================================================
-- FinSight AI — SQLite Database Schema
-- =============================================================================
-- Database file: data/finsight_ai.db  (relative to the stock_ai/ package root)
-- Configured via:  DATABASE_URL=sqlite:///data/finsight_ai.db  in .env
--
-- This schema is applied automatically on startup by:
--   database/connection.py  ->  init_db()  ->  Base.metadata.create_all()
--
-- You can also apply it manually with the SQLite CLI:
--   sqlite3 stock_ai/data/finsight_ai.db < stock_ai/database/create_db.sql
--
-- NOTE: SQLite does not support:
--   - Multiple databases  (no CREATE DATABASE / USE)
--   - CHARSET / ENGINE / COLLATE clauses  (omitted)
--   - Native JSON column type  (stored as TEXT — works fine with Python)
-- =============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;


-- =============================================================================
-- 1. MARKET DATA TABLES
-- =============================================================================

-- 1-a. instruments — master list of tradable symbols
CREATE TABLE IF NOT EXISTS instruments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    exchange      TEXT    NOT NULL,
    company_name  TEXT,
    isin          TEXT,
    sector        TEXT,
    active        INTEGER DEFAULT 1,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE (symbol, exchange)
);


-- 1-b. market_prices — daily OHLCV bars
CREATE TABLE IF NOT EXISTS market_prices (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id  INTEGER NOT NULL,
    trading_date   TEXT    NOT NULL,
    open_price     REAL    NOT NULL,
    high_price     REAL    NOT NULL,
    low_price      REAL    NOT NULL,
    close_price    REAL    NOT NULL,
    volume         INTEGER,
    turnover       REAL,
    created_at     TEXT    DEFAULT (datetime('now')),
    UNIQUE (instrument_id, trading_date),
    FOREIGN KEY (instrument_id) REFERENCES instruments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_market_prices_instrument_date
    ON market_prices(instrument_id, trading_date);


-- =============================================================================
-- 2. ML MODEL TABLES
-- =============================================================================

-- 2-a. model_versions — registry of trained model artefacts
CREATE TABLE IF NOT EXISTS model_versions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name        TEXT NOT NULL,
    version           TEXT NOT NULL,
    training_dataset  TEXT,
    training_date     TEXT DEFAULT (datetime('now')),
    metrics_json      TEXT,
    status            TEXT DEFAULT 'TRAINED',
    deployed_at       TEXT
);


-- 2-b. predictions — per-symbol directional probability forecasts
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id       INTEGER,
    prediction_time     TEXT    NOT NULL,
    horizon_days        INTEGER NOT NULL,
    probability_down    REAL,
    probability_neutral REAL,
    probability_up      REAL,
    predicted_class     INTEGER,
    model_version_id    INTEGER,
    created_at          TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (instrument_id)    REFERENCES instruments(id)    ON DELETE SET NULL,
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_instrument_time
    ON predictions(instrument_id, prediction_time);


-- =============================================================================
-- 3. NEWS INTELLIGENCE ENGINE TABLES
-- =============================================================================

-- 3-a. news_articles — raw ingested articles (deduplicated by SHA-256 hash)
CREATE TABLE IF NOT EXISTS news_articles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT    NOT NULL,
    summary       TEXT,
    content       TEXT,
    source        TEXT,
    url           TEXT,
    published_at  TEXT,
    symbol        TEXT,
    sector        TEXT,
    language      TEXT    DEFAULT 'en',
    content_hash  TEXT    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE (content_hash)
);

CREATE INDEX IF NOT EXISTS idx_news_articles_symbol_time ON news_articles(symbol, published_at);
CREATE INDEX IF NOT EXISTS idx_news_articles_time        ON news_articles(published_at);


-- 3-b. news_analysis — sentiment / impact scores per article
CREATE TABLE IF NOT EXISTS news_analysis (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id        INTEGER NOT NULL,
    sentiment      REAL,
    confidence     REAL,
    importance     REAL,
    event_type     TEXT,
    impact         REAL,
    method         TEXT,
    positive_hits  INTEGER DEFAULT 0,
    negative_hits  INTEGER DEFAULT 0,
    keywords_json  TEXT,
    created_at     TEXT    DEFAULT (datetime('now')),
    UNIQUE (news_id),
    FOREIGN KEY (news_id) REFERENCES news_articles(id) ON DELETE CASCADE
);


-- 3-c. news_feature_snapshots — aggregated feature vectors for ML pipeline
CREATE TABLE IF NOT EXISTS news_feature_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,
    feature_time  TEXT NOT NULL,
    features_json TEXT NOT NULL,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_news_features_symbol_time
    ON news_feature_snapshots(symbol, feature_time);


-- =============================================================================
-- 4. AUTHENTICATION & USER MANAGEMENT  (also auto-created by SQLAlchemy ORM)
-- =============================================================================

-- 4-a. users — registered accounts
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT    NOT NULL UNIQUE,
    password_hash  TEXT    NOT NULL,
    salt           TEXT    NOT NULL,
    role           TEXT    NOT NULL DEFAULT 'user',
    created_at     TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);


-- 4-b. user_sessions — persistent bearer-token sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    token       TEXT    NOT NULL UNIQUE,
    user_id     INTEGER NOT NULL,
    username    TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    expires_at  TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_token   ON user_sessions(token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);


-- 4-c. allowed_symbols — curated benchmark whitelist for standard users
CREATE TABLE IF NOT EXISTS allowed_symbols (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL UNIQUE,
    display_name  TEXT    NOT NULL,
    exchange      TEXT    NOT NULL,
    sector        TEXT,
    is_active     INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_allowed_symbols_symbol ON allowed_symbols(symbol);


-- 4-d. user_activity_logs — immutable audit trail
CREATE TABLE IF NOT EXISTS user_activity_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER,
    username   TEXT    NOT NULL DEFAULT 'guest',
    symbol     TEXT    NOT NULL,
    action     TEXT    NOT NULL,
    details    TEXT,
    timestamp  TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_user_activity_logs_user_id  ON user_activity_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_user_activity_logs_symbol    ON user_activity_logs(symbol);
CREATE INDEX IF NOT EXISTS idx_user_activity_logs_timestamp ON user_activity_logs(timestamp);


-- =============================================================================
-- 5. SEED DATA
-- =============================================================================
-- The app's init_db() (database/connection.py) seeds the admin row on first
-- boot using Python's secrets.token_hex() for fresh salts + SHA-256 hashes.
-- Those rows take precedence over these placeholders.
--
-- These inserts are only meaningful when you run this file manually via the
-- SQLite CLI BEFORE starting the app for the first time.
--
-- Password:  admin -> admin123
-- WARNING: Change credentials immediately after first login in production!
-- Users can only be created by the Administrator via the Admin Panel.
-- =============================================================================

-- 5-a. Default admin  (placeholder hash — app will re-seed with correct hash)
INSERT OR IGNORE INTO users (username, password_hash, salt, role, created_at)
VALUES (
    'admin',
    'placeholder_run_app_to_generate_real_hash',
    'placeholder_salt',
    'admin',
    datetime('now')
);

-- 5-c. Curated 6-symbol allowed dataset (mirrors CURATED_ALLOWED_DATASETS in connection.py)
INSERT OR IGNORE INTO allowed_symbols (symbol, display_name, exchange, sector, is_active)
VALUES
    ('TATAMOTORS.NS', 'Tata Motors Ltd',        'NSE',    'Automobile',         1),
    ('INFY.NS',       'Infosys Ltd',             'NSE',    'IT Consulting',       1),
    ('RELIANCE.NS',   'Reliance Industries',      'NSE',    'Energy & Telecom',    1),
    ('HDFCBANK.NS',   'HDFC Bank Ltd',            'NSE',    'Banking & Finance',   1),
    ('MSFT',          'Microsoft Corporation',    'NASDAQ', 'Cloud & Software',    1),
    ('NVDA',          'NVIDIA Corporation',       'NASDAQ', 'Semiconductors & AI', 1);


-- =============================================================================
-- Done. Start the app — init_db() will finalize seeding with correct hashes.
-- =============================================================================
