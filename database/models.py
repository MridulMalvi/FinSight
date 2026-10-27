"""
SQLAlchemy ORM models for FinSight AI Authentication,
Allowed Dataset Curation, and User Activity Audit Logging.
"""
import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """Registered user account."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    salt = Column(String(64), nullable=False)
    role = Column(String(20), default="user", nullable=False)   # "admin" | "user"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    activities = relationship("UserActivity", back_populates="user", cascade="all, delete-orphan")


class AllowedSymbol(Base):
    """Curated list of stock symbols that standard users are permitted to analyze."""
    __tablename__ = "allowed_symbols"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), unique=True, index=True, nullable=False)
    display_name = Column(String(100), nullable=False)
    exchange = Column(String(20), nullable=False)
    sector = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)


class UserSession(Base):
    """Persistent bearer-token session (survives server restarts)."""
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(128), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    username = Column(String(50), nullable=False)
    role = Column(String(20), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class UserActivity(Base):
    """Audit log of every stock analysis performed by a user."""
    __tablename__ = "user_activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    username = Column(String(50), default="guest", nullable=False)
    symbol = Column(String(50), index=True, nullable=False)
    action = Column(String(50), nullable=False)   # "stock_analysis", "news_fetch", etc.
    details = Column(Text, nullable=True)          # JSON snapshot of result
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    user = relationship("User", back_populates="activities")
