"""Where graded decisions are kept. MemoryGradeStore is for tests; SqlGradeStore (sql_store.py)
is the real one. Both follow the same contract (see tests/grade_store_contract.py)."""

import threading
from dataclasses import asdict

from .stats import aggregate
from .types import GradedDecision, SkillTotals


class MemoryGradeStore:
    def __init__(self) -> None:
        self._rows: list[tuple[int, int, GradedDecision]] = []  # (game_ref, user_id, decision)
        self._lock = threading.Lock()

    def save(self, game_ref: int, user_id: int, decisions: list[GradedDecision]) -> None:
        """Store a player's graded decisions for one match. Saving again REPLACES them, so a
        retry can never double count."""
        with self._lock:
            self._rows = [r for r in self._rows if not (r[0] == game_ref and r[1] == user_id)]
            self._rows.extend((game_ref, user_id, d) for d in decisions)

    def decisions_for(self, user_id: int, limit: int = 50, offset: int = 0) -> list[dict]:
        """A player's graded decisions, newest first."""
        with self._lock:
            mine = [(game_ref, d) for game_ref, uid, d in self._rows if uid == user_id]
        mine.reverse()
        return [{"game_id": game_ref, **asdict(d)} for game_ref, d in mine[offset:offset + limit]]

    def skill_totals(self) -> dict[int, SkillTotals]:
        with self._lock:
            rows = [(uid, d) for _game_ref, uid, d in self._rows]
        return aggregate(rows)
