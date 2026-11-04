"""Shared FastAPI dependencies."""

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User
from app.services.sessions import get_user_for_token


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """The logged-in user, or HTTP 401. Use this on every protected route."""
    token = request.cookies.get(settings.session_cookie_name)
    user = get_user_for_token(db, token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
