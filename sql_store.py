"""SqlGradeStore: graded decisions in the database (same contract as MemoryGradeStore)."""

from collections.abc import Callable

from sqlalchemy import case, delete, func, select
from sqlalchemy.orm import Session

from app.models import GradedDecisionRecord as Row

from .types import GradedDecision, SkillTotals

FIELDS = (
    "hand_number", "decision_index", "seat", "decision_kind", "own_value", "action_kind",
    "action_amount", "best_kind", "best_amount", "ev_taken", "ev_best", "regret", "hand_net", "counts_for_ev",
)


class SqlGradeStore:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def save(self, game_ref: int, user_id: int, decisions: list[GradedDecision]) -> None:
        """Store a player's graded decisions for one match, REPLACING any saved earlier for the
        same match (so a retry cannot double count). One transaction."""
        with self._session_factory() as db:
            try:
                db.execute(
                    delete(Row).where(Row.game_id == game_ref, Row.user_id == user_id),
                    execution_options={"synchronize_session": False},
                )
                db.add_all(
                    Row(game_id=game_ref, user_id=user_id, **{f: getattr(d, f) for f in FIELDS})
                    for d in decisions
                )
                db.commit()
            except Exception:
                db.rollback()
                raise

    def decisions_for(self, user_id: int, limit: int = 50, offset: int = 0) -> list[dict]:
        """A player's graded decisions, newest first."""
        with self._session_factory() as db:
            rows = db.execute(
                select(Row).where(Row.user_id == user_id).order_by(Row.id.desc()).limit(limit).offset(offset)
            ).scalars().all()
            return [{"game_id": r.game_id, **{f: getattr(r, f) for f in FIELDS}} for r in rows]

    def skill_totals(self) -> dict[int, SkillTotals]:
        counted = Row.counts_for_ev.is_(True)
        with self._session_factory() as db:
            rows = db.execute(
                select(
                    Row.user_id,
                    func.count(),
                    func.sum(Row.regret),
                    func.sum(case((counted, 1), else_=0)),
                    func.sum(case((counted, Row.ev_taken), else_=0.0)),
                    func.sum(case((counted, Row.hand_net), else_=0)),
                    func.sum(case((counted, Row.ev_best), else_=0.0)),
                ).group_by(Row.user_id)
            ).all()
        return {
            user_id: SkillTotals(user_id, int(hands), int(decisions), float(ev), float(regret), int(actual), float(potential))
            for user_id, decisions, regret, hands, ev, actual, potential in rows
        }
