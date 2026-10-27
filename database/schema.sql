-- =============================================================================
-- FinSight AI — MySQL-Compatible Schema
-- =============================================================================
-- Run this manually if you need the legacy "raw" tables.
-- The ORM models in database/models.py (users, user_sessions,
-- allowed_symbols, user_activity_logs) are created automatically via
-- SQLAlchemy Base.metadata.create_all() on startup.
-- =============================================================================

CREATE TABLE IF NOT EXISTS instruments (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    symbol        VARCHAR(50)  NOT NULL,
    exchange      VARCHAR(20)  NOT NULL,
    company_name  VARCHAR(255),
    isin          VARCHAR(20),
    sector        VARCHAR(100),
    active        TINYINT(1)   DEFAULT 1,
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_symbol_exchange (symbol, exchange)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE IF NOT EXISTS market_prices (
    id             BIGINT AUTO_INCREMENT PRIMARY KEY,
    instrument_id  INT          NOT NULL,
    trading_date   DATE         NOT NULL,
    open_price     DECIMAL(18,4) NOT NULL,
    high_price     DECIMAL(18,4) NOT NULL,
    low_price      DECIMAL(18,4) NOT NULL,
    close_price    DECIMAL(18,4) NOT NULL,
    volume         BIGINT,
    turnover       DECIMAL(24,4),
    created_at     DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_instrument_date (instrument_id, trading_date),
    CONSTRAINT fk_mp_instrument FOREIGN KEY (instrument_id) REFERENCES instruments(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_market_prices_instrument_date ON market_prices(instrument_id, trading_date);


CREATE TABLE IF NOT EXISTS model_versions (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    model_name        VARCHAR(100) NOT NULL,
    version           VARCHAR(50)  NOT NULL,
    training_dataset  TEXT,
    training_date     DATETIME     DEFAULT CURRENT_TIMESTAMP,
    metrics_json      JSON,
    status            VARCHAR(30)  DEFAULT 'TRAINED',
    deployed_at       DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    instrument_id       INT,
    prediction_time     DATETIME     NOT NULL,
    horizon_days        INT          NOT NULL,
    probability_down    DECIMAL(10,8),
    probability_neutral DECIMAL(10,8),
    probability_up      DECIMAL(10,8),
    predicted_class     INT,
    model_version_id    INT,
    created_at          DATETIME     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_pred_instrument    FOREIGN KEY (instrument_id)    REFERENCES instruments(id),
    CONSTRAINT fk_pred_model_version FOREIGN KEY (model_version_id) REFERENCES model_versions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- News Intelligence Engine
CREATE TABLE IF NOT EXISTS news_articles (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    title         TEXT         NOT NULL,
    summary       TEXT,
    content       LONGTEXT,
    source        VARCHAR(255),
    url           TEXT,
    published_at  DATETIME,
    symbol        VARCHAR(50),
    sector        VARCHAR(100),
    language      VARCHAR(10)  DEFAULT 'en',
    content_hash  CHAR(64)     NOT NULL,
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_content_hash (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_news_articles_symbol_time ON news_articles(symbol, published_at);
CREATE INDEX idx_news_articles_time        ON news_articles(published_at);


CREATE TABLE IF NOT EXISTS news_analysis (
    id             BIGINT AUTO_INCREMENT PRIMARY KEY,
    news_id        BIGINT        NOT NULL,
    sentiment      DECIMAL(10,8),
    confidence     DECIMAL(10,8),
    importance     DECIMAL(10,8),
    event_type     VARCHAR(50),
    impact         DECIMAL(10,8),
    method         VARCHAR(30),
    positive_hits  INT           DEFAULT 0,
    negative_hits  INT           DEFAULT 0,
    keywords_json  JSON,
    created_at     DATETIME      DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_news_analysis_news_id (news_id),
    CONSTRAINT fk_na_news FOREIGN KEY (news_id) REFERENCES news_articles(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE IF NOT EXISTS news_feature_snapshots (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol        VARCHAR(50),
    feature_time  DATETIME  NOT NULL,
    features_json JSON      NOT NULL,
    created_at    DATETIME  DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_news_features_symbol_time ON news_feature_snapshots(symbol, feature_time);
