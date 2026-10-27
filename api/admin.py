"""
Admin & User Panel API
-----------------------
Provides:
  - Admin: Audit logs of all user analysis activities
  - Admin: User list with stats
  - User: Personal history of own analyses
  - Public: Allowed curated symbol list
"""

import json
import logging
import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.connection import get_db
from database.models import User, UserActivity, AllowedSymbol
from api.auth import require_admin, require_user, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin & User Panel"])

SESSION_WINDOW_HOURS = 2
USER_ANALYSIS_LIMIT = 7  # maximum analyses for regular (non-admin) users


# ─── Helpers ─────────────────────────────────────────────────────────────────

def log_activity(
    db: Session,
    symbol: str,
    action: str,
    details: dict,
    session: Optional[dict] = None,
):
    """Log a user activity to the audit database.
    
    Consolidates technical multi-horizon predictions (1W, 15D, 1M) and news-adjusted
    predictions performed for the same stock in a single session into one single row.
    """
    try:
        username = session["username"] if session else "guest"
        user_id = session["user_id"] if session else None
        sym_upper = symbol.upper()

        if action == "stock_analysis":
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=SESSION_WINDOW_HOURS)
            existing = (
                db.query(UserActivity)
                .filter(
                    UserActivity.username == username,
                    UserActivity.symbol == sym_upper,
                    UserActivity.action == action,
                    UserActivity.timestamp >= cutoff,
                )
                .order_by(UserActivity.timestamp.desc())
                .first()
            )
            if existing:
                try:
                    existing_details = json.loads(existing.details) if existing.details else {}
                except Exception:
                    existing_details = {}

                # 1. Merge horizons if provided and non-empty
                new_horizons = details.get("horizons")
                if new_horizons and isinstance(new_horizons, dict) and len(new_horizons) > 0:
                    merged_horizons = existing_details.get("horizons")
                    if not isinstance(merged_horizons, dict):
                        merged_horizons = {}
                    merged_horizons.update(new_horizons)
                    existing_details["horizons"] = merged_horizons

                # 2. Merge news_adj if provided
                new_news_adj = details.get("news_adj")
                if new_news_adj:
                    existing_details["news_adj"] = new_news_adj

                # 3. Merge metadata fields
                for key in ("close", "currency_symbol", "company_name", "data_source", "news_source", "source"):
                    val = details.get(key)
                    if val is not None and val != "":
                        existing_details[key] = val

                existing.details = json.dumps(existing_details, default=str)
                existing.timestamp = datetime.datetime.utcnow()
                db.commit()
                db.refresh(existing)
                return existing

        entry = UserActivity(
            user_id=user_id,
            username=username,
            symbol=sym_upper,
            action=action,
            details=json.dumps(details, default=str),
            timestamp=datetime.datetime.utcnow(),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry
    except Exception as e:
        db.rollback()
        logger.warning("[AuditLog] Failed to log activity: %s", e)


def check_analysis_quota(db: Session, session: Optional[dict], symbol: str):
    """Enforce analysis quota (max 7) for regular users.
    
    If the user has already analyzed this stock in the current session (last 2 hours),
    allow the request through without counting it as a new transaction, since it will
    be merged into their existing analysis record.
    """
    if not session:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in to analyze stocks.")
    if session.get("role") == "admin":
        return

    sym_upper = symbol.upper()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=SESSION_WINDOW_HOURS)
    existing = (
        db.query(UserActivity)
        .filter(
            UserActivity.username == session["username"],
            UserActivity.symbol == sym_upper,
            UserActivity.action == "stock_analysis",
            UserActivity.timestamp >= cutoff,
        )
        .first()
    )
    if existing:
        return

    used = (
        db.query(func.count(UserActivity.id))
        .filter(
            UserActivity.username == session["username"],
            UserActivity.action == "stock_analysis",
        )
        .scalar() or 0
    )
    if used >= USER_ANALYSIS_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "QUOTA_EXHAUSTED",
                "message": (
                    f"You have reached your analysis limit of {USER_ANALYSIS_LIMIT} stocks. "
                    f"Contact the admin to increase your quota."
                ),
                "used": used,
                "limit": USER_ANALYSIS_LIMIT,
                "remaining": 0,
            },
        )


def consolidate_session_activities(db: Session, window_hours: int = 2):
    """Consolidate any existing split analysis records within the same session into a single row."""
    try:
        rows = (
            db.query(UserActivity)
            .filter(UserActivity.action == "stock_analysis")
            .order_by(UserActivity.username, UserActivity.symbol, UserActivity.timestamp.asc())
            .all()
        )
        groups = {}
        for r in rows:
            groups.setdefault((r.username, r.symbol), []).append(r)

        modified = False
        for key, g in groups.items():
            if len(g) < 2:
                continue
            curr = g[0]
            try:
                curr_details = json.loads(curr.details) if curr.details else {}
            except Exception:
                curr_details = {}

            for next_row in g[1:]:
                diff = (next_row.timestamp - curr.timestamp).total_seconds()
                if 0 <= diff <= (window_hours * 3600):
                    try:
                        next_details = json.loads(next_row.details) if next_row.details else {}
                    except Exception:
                        next_details = {}

                    # Merge horizons
                    if next_details.get("horizons"):
                        curr_horizons = curr_details.get("horizons") or {}
                        if isinstance(curr_horizons, dict):
                            curr_horizons.update(next_details["horizons"])
                            curr_details["horizons"] = curr_horizons
                    # Merge news_adj
                    if next_details.get("news_adj"):
                        curr_details["news_adj"] = next_details["news_adj"]
                    # Merge metadata
                    for k in ("close", "currency_symbol", "company_name", "data_source", "news_source", "source"):
                        if next_details.get(k):
                            curr_details[k] = next_details[k]

                    curr.details = json.dumps(curr_details, default=str)
                    curr.timestamp = max(curr.timestamp, next_row.timestamp)
                    db.delete(next_row)
                    modified = True
                else:
                    curr = next_row
                    try:
                        curr_details = json.loads(curr.details) if curr.details else {}
                    except Exception:
                        curr_details = {}

        if modified:
            db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("[AuditLog] Failed to consolidate activities: %s", e)


# ─── Public: Allowed Curated Datasets ─────────────────────────────────────────

@router.get("/api/v1/allowed-symbols")
def get_allowed_symbols(db: Session = Depends(get_db)):
    """Returns the 8 curated allowed benchmark datasets that users can analyze."""
    symbols = db.query(AllowedSymbol).filter(AllowedSymbol.is_active == True).all()
    return {
        "allowed_symbols": [
            {
                "symbol": s.symbol,
                "display_name": s.display_name,
                "exchange": s.exchange,
                "sector": s.sector,
            }
            for s in symbols
        ],
        "count": len(symbols),
    }


# ─── User: Personalized Workspace Stocks ──────────────────────────────────────

@router.get("/api/v1/user/workspace-stocks")
def get_user_workspace_stocks(
    limit: int = Query(default=8, le=8),
    session: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Returns the distinct stocks a logged-in user has previously analyzed.

    Used to populate the "Your Stock Workspace" grid with stocks personal to
    this user instead of showing everyone the same admin-curated benchmark list.

    Returns newest-first, deduplicated by symbol, capped at `limit` (max 8).
    Metadata (display_name, exchange, sector) is sourced from AllowedSymbol
    if the stock is a benchmark, otherwise parsed from the stored activity details.
    """
    # Fetch all stock_analysis activities for this user, newest first
    activities = (
        db.query(UserActivity)
        .filter(
            UserActivity.username == session["username"],
            UserActivity.action == "stock_analysis",
        )
        .order_by(UserActivity.timestamp.desc())
        .all()
    )

    # Build an AllowedSymbol lookup for metadata enrichment
    allowed_map: dict = {}
    try:
        allowed_rows = db.query(AllowedSymbol).all()
        allowed_map = {row.symbol.upper(): row for row in allowed_rows}
    except Exception:
        pass

    # Deduplicate by symbol (preserve newest-first order), cap at limit
    seen: set = set()
    workspace: list = []
    for act in activities:
        sym_upper = act.symbol.upper()
        if sym_upper in seen:
            continue
        seen.add(sym_upper)

        # Prefer AllowedSymbol metadata; fall back to activity details JSON
        if sym_upper in allowed_map:
            row = allowed_map[sym_upper]
            entry = {
                "symbol": row.symbol,
                "display_name": row.display_name or row.symbol,
                "exchange": row.exchange or "NSE",
                "sector": row.sector or "Equities",
                "is_benchmark": True,
            }
        else:
            # Try to extract company_name from stored details JSON
            details: dict = {}
            if act.details:
                try:
                    details = json.loads(act.details)
                except Exception:
                    pass
            company_name = details.get("company_name") or details.get("company") or act.symbol
            entry = {
                "symbol": act.symbol,
                "display_name": company_name,
                "exchange": "YF",
                "sector": "Equities",
                "is_benchmark": False,
            }

        workspace.append(entry)
        if len(workspace) >= limit:
            break

    return {
        "username": session["username"],
        "count": len(workspace),
        "workspace_stocks": workspace,
    }


# ─── User: Personal History ────────────────────────────────────────────────────

@router.get("/api/v1/user/history")
def get_user_history(
    limit: int = Query(default=50, le=200),
    session: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Returns the logged-in user's past analysis activities (newest first)."""
    consolidate_session_activities(db)
    activities = (
        db.query(UserActivity)
        .filter(UserActivity.username == session["username"])
        .order_by(UserActivity.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "username": session["username"],
        "total": len(activities),
        "activities": [
            {
                "id": a.id,
                "symbol": a.symbol,
                "action": a.action,
                "details": json.loads(a.details) if a.details else {},
                "timestamp": a.timestamp.isoformat() + "Z",
            }
            for a in activities
        ],
    }


# ─── User: Analysis Quota ──────────────────────────────────────────────────────

@router.get("/api/v1/user/quota")
def get_user_quota(
    session: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Returns the analysis quota for the current user.
    Admin users have unlimited access. Regular users are capped at USER_ANALYSIS_LIMIT.
    """
    consolidate_session_activities(db)
    is_admin = session.get("role") == "admin"
    if is_admin:
        return {
            "username": session["username"],
            "role": "admin",
            "limit": None,
            "used": None,
            "remaining": None,
            "exhausted": False,
            "unlimited": True,
        }
    used = (
        db.query(func.count(UserActivity.id))
        .filter(
            UserActivity.username == session["username"],
            UserActivity.action == "stock_analysis",
        )
        .scalar() or 0
    )
    remaining = max(0, USER_ANALYSIS_LIMIT - used)
    return {
        "username": session["username"],
        "role": session.get("role", "user"),
        "limit": USER_ANALYSIS_LIMIT,
        "used": used,
        "remaining": remaining,
        "exhausted": used >= USER_ANALYSIS_LIMIT,
        "unlimited": False,
    }


# ─── Admin: Audit Logs ────────────────────────────────────────────────────────

@router.get("/api/v1/admin/audit-logs")
def get_audit_logs(
    limit: int = Query(default=100, le=500),
    username: Optional[str] = Query(default=None, description="Filter by username"),
    symbol: Optional[str] = Query(default=None, description="Filter by stock symbol"),
    session: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin only: Returns chronological audit log of all user analyses."""
    consolidate_session_activities(db)
    query = db.query(UserActivity).order_by(UserActivity.timestamp.desc())
    if username:
        query = query.filter(UserActivity.username == username)
    if symbol:
        query = query.filter(UserActivity.symbol == symbol.upper())
    activities = query.limit(limit).all()

    return {
        "total_fetched": len(activities),
        "filters": {"username": username, "symbol": symbol},
        "audit_logs": [
            {
                "id": a.id,
                "username": a.username,
                "symbol": a.symbol,
                "action": a.action,
                "details": json.loads(a.details) if a.details else {},
                "timestamp": a.timestamp.isoformat() + "Z",
            }
            for a in activities
        ],
    }


# ─── Admin: User List ─────────────────────────────────────────────────────────

@router.get("/api/v1/admin/users")
def get_all_users(
    session: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin only: Lists all registered users with their activity stats."""
    users = db.query(User).order_by(User.created_at.asc()).all()

    # Activity count per user
    counts = dict(
        db.query(UserActivity.username, func.count(UserActivity.id))
        .group_by(UserActivity.username)
        .all()
    )
    # Last activity per user
    last_active = dict(
        db.query(UserActivity.username, func.max(UserActivity.timestamp))
        .group_by(UserActivity.username)
        .all()
    )

    return {
        "total_users": len(users),
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "role": u.role,
                "created_at": u.created_at.isoformat() + "Z",
                "total_analyses": counts.get(u.username, 0),
                "last_active": (
                    last_active[u.username].isoformat() + "Z"
                    if last_active.get(u.username)
                    else None
                ),
            }
            for u in users
        ],
    }


# ─── Admin: Dashboard KPIs ────────────────────────────────────────────────────

@router.get("/api/v1/admin/dashboard")
def get_dashboard_kpis(
    session: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin only: Summary KPI metrics for the admin dashboard panel."""
    total_users = db.query(func.count(User.id)).scalar()
    total_analyses = db.query(func.count(UserActivity.id)).scalar()

    # Most analyzed symbol
    top_symbol_row = (
        db.query(UserActivity.symbol, func.count(UserActivity.id).label("cnt"))
        .group_by(UserActivity.symbol)
        .order_by(func.count(UserActivity.id).desc())
        .first()
    )
    most_analyzed = top_symbol_row[0] if top_symbol_row else "N/A"
    most_analyzed_count = top_symbol_row[1] if top_symbol_row else 0

    # Most active user
    top_user_row = (
        db.query(UserActivity.username, func.count(UserActivity.id).label("cnt"))
        .group_by(UserActivity.username)
        .order_by(func.count(UserActivity.id).desc())
        .first()
    )
    most_active_user = top_user_row[0] if top_user_row else "N/A"
    most_active_count = top_user_row[1] if top_user_row else 0

    # Today's analyses
    today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = (
        db.query(func.count(UserActivity.id))
        .filter(UserActivity.timestamp >= today_start)
        .scalar()
    )

    return {
        "total_users": total_users,
        "total_analyses": total_analyses,
        "today_analyses": today_count,
        "most_analyzed_stock": most_analyzed,
        "most_analyzed_count": most_analyzed_count,
        "most_active_user": most_active_user,
        "most_active_user_count": most_active_count,
    }
