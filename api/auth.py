"""
Authentication API
------------------
Provides login, register, logout, and /me endpoints.
Uses simple SHA-256 hashed passwords with per-user salt.
Bearer tokens are stored in the DB (UserSession table) — survive server restarts.
"""

import logging
import secrets
import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.connection import SessionLocal, hash_password, verify_password, get_db
from database.models import User, UserSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

SESSION_TTL_HOURS = 72   # 3-day sessions — survive --reload restarts


# ─── Request / Response Schemas ─────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    role: str
    message: str


class UserInfo(BaseModel):
    user_id: int
    username: str
    role: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_token() -> str:
    return secrets.token_urlsafe(32)


def _create_session(user: User, db: Session) -> str:
    token = _make_token()
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=SESSION_TTL_HOURS)
    session = UserSession(
        token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,
        expires_at=expires_at,
    )
    db.add(session)
    db.commit()
    return token


def get_current_user(authorization: Optional[str] = Header(None, alias="Authorization")) -> Optional[dict]:
    """Dependency: Returns session dict if valid bearer token is present, else None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    # Look up in DB
    db = SessionLocal()
    try:
        session = db.query(UserSession).filter(UserSession.token == token).first()
        if not session:
            return None
        if datetime.datetime.utcnow() > session.expires_at:
            db.delete(session)
            db.commit()
            return None
        return {
            "user_id": session.user_id,
            "username": session.username,
            "role": session.role,
        }
    except Exception:
        return None
    finally:
        db.close()


def require_user(session: Optional[dict] = Depends(get_current_user)) -> dict:
    """Dependency: Require any authenticated user (user or admin)."""
    if not session:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    return session


def require_admin(session: Optional[dict] = Depends(get_current_user)) -> dict:
    """Dependency: Require admin role."""
    if not session:
        raise HTTPException(status_code=401, detail="Authentication required.")
    if session.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return session


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login with username and password. Returns a bearer token."""
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not verify_password(request.password, user.salt, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = _create_session(user, db)
    logger.info("[Auth] Login: %s (role=%s)", user.username, user.role)
    return AuthResponse(
        token=token,
        username=user.username,
        role=user.role,
        message=f"Welcome back, {user.username}!"
    )


@router.post("/register", response_model=AuthResponse)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
    session: dict = Depends(require_admin),
):
    """Admin-only: Create a new user account.
    
    Self-registration is disabled. Only admins can create user accounts.
    The created account always receives the 'user' role.
    """
    if len(request.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = db.query(User).filter(User.username == request.username).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Username '{request.username}' is already taken.")

    pwd_hash, salt = hash_password(request.password)
    user = User(
        username=request.username,
        password_hash=pwd_hash,
        salt=salt,
        role="user"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info("[Auth] Admin '%s' created new user: %s", session.get("username"), user.username)
    return AuthResponse(
        token="",  # No auto-login token for admin-created accounts
        username=user.username,
        role=user.role,
        message=f"User '{user.username}' created successfully."
    )


@router.get("/me")
def get_me(session: dict = Depends(require_user)):
    """Get info about the currently authenticated user."""
    return {
        "user_id": session["user_id"],
        "username": session["username"],
        "role": session["role"],
    }


@router.post("/logout")
def logout(authorization: Optional[str] = Header(None, alias="Authorization"), db: Session = Depends(get_db)):
    """Invalidate the current session token."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        session = db.query(UserSession).filter(UserSession.token == token).first()
        if session:
            db.delete(session)
            db.commit()
    return {"message": "Logged out successfully."}
