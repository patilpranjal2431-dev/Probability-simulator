"""The two leaderboards. Both need the login cookie."""

from fastapi import APIRouter, Depends, Query, Request

from app.deps import get_current_user
from app.models import User

router = APIRouter(tags=["leaderboards"])


def present(row: dict) -> dict:
    """Round floats for display (the stored values stay exact)."""
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}


@router.get("/leaderboards")
def leaderboards(
    request: Request,
    min_hands: int = Query(1, ge=1, le=10_000, description="only rank players with at least this many counted hands"),
    limit: int = Query(50, ge=1, le=100),
    _user: User = Depends(get_current_user),
) -> dict:
    """{"skill": [...], "coins": [...]}: ranked by cumulative EV (skill) and by net coins (luck + skill)."""
    boards = request.app.state.standings.both(min_hands, limit)
    return {name: [present(row) for row in rows] for name, rows in boards.items()}
