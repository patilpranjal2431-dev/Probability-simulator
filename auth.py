"""Register, login and logout."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User
from app.schemas import LoginIn, RegisterIn, UserOut
from app.security import burn_time, hash_password, verify_password
from app.services.sessions import (
    create_session,
    delete_expired_sessions,
    delete_session,
)
from app.services.wallet import create_user

router = APIRouter(prefix="/auth", tags=["auth"])

INVALID_LOGIN = "Invalid username or password"  # same text for unknown user and wrong password


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,  # JavaScript cannot read it, which limits the damage of XSS
        samesite="lax",  # not sent on cross-site POSTs, which blocks most CSRF
        secure=settings.cookie_secure,
        path="/",
    )


@router.post("/register", status_code=201, response_model=UserOut)
def register(body: RegisterIn, response: Response, db: Session = Depends(get_db)) -> UserOut:
    """Create an account (with 1,000 starting coins) and log the new user in."""
    try:
        user = create_user(db, body.username, hash_password(body.password))
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username already taken") from None

    delete_expired_sessions(db)
    token = create_session(db, user.id)
    db.commit()
    _set_session_cookie(response, token)
    return UserOut(id=user.id, username=user.username)


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, response: Response, db: Session = Depends(get_db)) -> UserOut:
    """Check the password and start a new session (a fresh token on every login)."""
    user = db.execute(select(User).where(User.username == body.username)).scalar_one_or_none()

    if user is not None and not user.is_bot:
        ok = verify_password(user.password_hash, body.password)
    else:
        burn_time(body.password)
        ok = False
    if not ok:
        raise HTTPException(status_code=401, detail=INVALID_LOGIN)

    delete_expired_sessions(db)
    token = create_session(db, user.id)
    db.commit()
    _set_session_cookie(response, token)
    return UserOut(id=user.id, username=user.username)


@router.post("/logout", status_code=204, response_class=Response)
def logout(request: Request, db: Session = Depends(get_db)) -> Response:
    """Revoke the session on the server and clear the cookie. Safe to call when logged out."""
    token = request.cookies.get(settings.session_cookie_name)
    if token:
        delete_session(db, token)
        db.commit()
    response = Response(status_code=204)
    response.delete_cookie(settings.session_cookie_name, path="/")
    return response
