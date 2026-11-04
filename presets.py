"""Ready-made bets, several taken from the project's own games."""

SIDES = 6  # Dice Duel uses a six-sided die


def dice_duel(roll: int) -> dict:
    """Holding `roll` against a uniformly random opponent roll, with the opponent calling any bet.
    The same numbers the engine reports as a hand's win/lose/tie chances."""
    return {
        "id": f"dice_duel_{roll}",
        "name": f"Dice Duel: you hold a {roll}",
        "description": "Even money against an opponent who always calls. A tie is a push.",
        "win": (roll - 1) / SIDES,
        "lose": (SIDES - roll) / SIDES,
        "odds": 1.0,
    }


PRESETS = [
    {"id": "coin_55", "name": "A slightly favourable coin", "description": "55% to win, even money.",
     "win": 0.55, "lose": 0.45, "odds": 1.0},
    {"id": "coin_fair", "name": "A fair coin", "description": "No edge, so the best stake is zero.",
     "win": 0.5, "lose": 0.5, "odds": 1.0},
    {"id": "long_shot", "name": "A long shot", "description": "20% to win, pays 5-to-1.",
     "win": 0.2, "lose": 0.8, "odds": 5.0},
    dice_duel(4), dice_duel(5), dice_duel(6),
    {**dice_duel(3), "description": "A losing hand: even money against an opponent who always calls has a negative edge."},
]
