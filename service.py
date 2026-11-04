"""Assemble the two leaderboards from the grade store and the ledger."""

from collections.abc import Callable

from .stats import Names, coin_ranking, skill_ranking, with_rank_gap


class Standings:
    def __init__(self, grades, ledger, directory: Callable[[], Names]) -> None:
        self.grades = grades  # skill_totals()
        self.ledger = ledger  # net_coin_rows()
        self.directory = directory  # () -> {user_id: (username, is_bot)}

    def both(self, min_hands: int = 1, limit: int = 50) -> dict[str, list[dict]]:
        """{"skill": [...], "coins": [...]}, each row also showing the player's rank on the other board."""
        names = self.directory()
        skill = skill_ranking(self.grades.skill_totals(), names, min_hands)
        coins = coin_ranking(self.ledger.net_coin_rows(), names)
        skill, coins = with_rank_gap(skill, coins)
        return {"skill": skill[:limit], "coins": coins[:limit]}
