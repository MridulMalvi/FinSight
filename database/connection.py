import os
import hashlib
import secrets
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.settings import BASE_DIR, DATABASE_URL

logger = logging.getLogger(__name__)

# SQLite path for guaranteed local zero-setup persistence
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH = DATA_DIR / "finsight_ai.db"
LEGACY_DB_PATH = DATA_DIR / "stock_ai.db"
if not SQLITE_DB_PATH.exists() and LEGACY_DB_PATH.exists():
    try:
        import shutil
        shutil.copy2(LEGACY_DB_PATH, SQLITE_DB_PATH)
        logger.info("[DB] Migrated database from %s to %s", LEGACY_DB_PATH.name, SQLITE_DB_PATH.name)
    except Exception as _e:
        logger.warning("[DB] Failed migrating legacy db: %s", _e)


def get_engine():
    """Configure engine: try MySQL if set, fall back to local SQLite."""
    target_url = DATABASE_URL

    if "mysql" in target_url:
        try:
            test_engine = create_engine(
                target_url,
                pool_pre_ping=True,
                pool_recycle=28800,  # recycle connections every 8 hours
                connect_args={"charset": "utf8mb4"},
            )
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("[DB] Connected to MySQL successfully.")
            return test_engine
        except Exception as e:
            logger.warning(
                "[DB] MySQL unavailable (%s). Falling back to local SQLite at %s",
                e,
                SQLITE_DB_PATH,
            )
            target_url = f"sqlite:///{SQLITE_DB_PATH}"

    if target_url.startswith("sqlite"):
        return create_engine(target_url, connect_args={"check_same_thread": False})
    return create_engine(target_url, pool_pre_ping=True)


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str, salt: str = None):
    """Hash a password with SHA-256 + per-user salt. Returns (hash, salt)."""
    if not salt:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return pwd_hash, salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verify a plain-text password against a stored hash+salt."""
    pwd_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(pwd_hash, expected_hash)


# Curated benchmark datasets (6 standard high-profile stocks)
CURATED_ALLOWED_DATASETS = [
    {"symbol": "TATAMOTORS.NS", "display_name": "Tata Motors Ltd",       "exchange": "NSE",    "sector": "Automobile"},
    {"symbol": "INFY.NS",       "display_name": "Infosys Ltd",            "exchange": "NSE",    "sector": "IT Consulting"},
    {"symbol": "RELIANCE.NS",   "display_name": "Reliance Industries",     "exchange": "NSE",    "sector": "Energy & Telecom"},
    {"symbol": "HDFCBANK.NS",   "display_name": "HDFC Bank Ltd",          "exchange": "NSE",    "sector": "Banking & Finance"},
    {"symbol": "MSFT",          "display_name": "Microsoft Corporation",   "exchange": "NASDAQ", "sector": "Cloud & Software"},
    {"symbol": "NVDA",          "display_name": "NVIDIA Corporation",      "exchange": "NASDAQ", "sector": "Semiconductors & AI"},
]


def init_db():
    """Initialize all ORM tables and seed default admin, user, and 6 allowed datasets."""
    from database.models import Base, User, AllowedSymbol, UserSession  # noqa: F401 — registers table

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # 1. Seed Admin
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            pwd_hash, salt = hash_password("admin123")
            db.add(User(username="admin", password_hash=pwd_hash, salt=salt, role="admin"))
            logger.info("[DB] Seeded admin account (username: admin, password: admin123)")


        # 3. Seed Allowed 6 Datasets & prune removed benchmarks
        curated_syms = {item["symbol"] for item in CURATED_ALLOWED_DATASETS}
        db.query(AllowedSymbol).filter(~AllowedSymbol.symbol.in_(curated_syms)).delete(synchronize_session=False)

        for item in CURATED_ALLOWED_DATASETS:
            entry = db.query(AllowedSymbol).filter(AllowedSymbol.symbol == item["symbol"]).first()
            if not entry:
                db.add(AllowedSymbol(
                    symbol=item["symbol"],
                    display_name=item["display_name"],
                    exchange=item["exchange"],
                    sector=item["sector"],
                    is_active=True,
                ))
            else:
                entry.display_name = item["display_name"]
                entry.exchange = item["exchange"]
                entry.sector = item["sector"]
                entry.is_active = True

        db.commit()
        logger.info("[DB] Database initialized and seeded successfully.")
    except Exception as e:
        db.rollback()
        logger.error("[DB] Error initializing database: %s", e)
    finally:
        db.close()
