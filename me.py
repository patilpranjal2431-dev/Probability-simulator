"""The logged-in user's own profile data. Every route here requires a session."""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Transaction, User
from app.schemas import MeOut, TransactionOut

router = APIRouter(prefix="/me", tags=["me"])


@router.get("", response_model=MeOut)
def me(user: User = Depends(get_current_user)) -> MeOut:
    """Username and current coin balance."""
    return MeOut(id=user.id, username=user.username, balance=user.wallet.balance)


@router.get("/transactions", response_model=list[TransactionOut])
def my_transactions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Transaction]:
    """Coin history, newest first. Only ever returns the caller's own wallet."""
    stmt = (
        select(Transaction)
        .where(Transaction.wallet_id == user.wallet.id)
        .order_by(Transaction.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.execute(stmt).scalars())


@router.get("/decisions")
def my_decisions(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
) -> list[dict]:
    """Your graded decisions, newest first: what you did, the best action, and the regret."""
    rows = request.app.state.grade_store.decisions_for(user.id, limit, offset)
    return [{k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()} for row in rows]
