"""Look up games by name."""

from .base import Game
from .dice_duel import DiceDuel
from .kuhn import Kuhn

GAMES: dict[str, type[Game]] = {"dice": DiceDuel, "kuhn": Kuhn}


def make_game(name: str) -> Game:
    if name not in GAMES:
        raise ValueError(f"unknown game {name!r}; choose from {', '.join(sorted(GAMES))}")
    return GAMES[name]()


def buy_in_limits(game: Game) -> tuple[int, int]:
    """Allowed buy-in per player: from 10 antes up to the game's standard stack."""
    return game.ante * 10, game.default_stack
