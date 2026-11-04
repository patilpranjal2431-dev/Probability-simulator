"""Presets: valid bets, and the Dice Duel ones equal what the real game engine reports."""

import random

import pytest

from app.bankroll.model import BetSpec
from app.bankroll.presets import PRESETS, dice_duel
from app.games.dice_duel import DiceDuel


def test_every_preset_is_a_valid_bet_with_a_unique_id():
    ids = [p["id"] for p in PRESETS]
    assert len(ids) == len(set(ids))
    for preset in PRESETS:
        spec = BetSpec(preset["win"], preset["lose"], preset["odds"])
        assert spec.win + spec.lose <= 1 + 1e-9
        assert preset["name"] and preset["description"]


@pytest.mark.parametrize("roll", range(1, 7))
def test_dice_duel_presets_equal_the_engines_own_odds(roll):
    """The engine reports (win, tie, lose) chances for a roll; the preset must be exactly those."""
    hand = DiceDuel().new_hand(random.Random(0), (1000, 1000), 0, deal=(roll, 1))
    decision = hand.decision()
    preset = dice_duel(roll)
    assert preset["win"] == pytest.approx(decision.win_prob)
    assert preset["lose"] == pytest.approx(decision.lose_prob)
    assert 1 - preset["win"] - preset["lose"] == pytest.approx(decision.tie_prob)


def test_the_presets_tell_the_story_they_claim_to():
    by_id = {p["id"]: BetSpec(p["win"], p["lose"], p["odds"]) for p in PRESETS}
    assert by_id["coin_55"].kelly_fraction() == pytest.approx(0.10)
    assert by_id["coin_fair"].kelly_fraction() == 0
    assert by_id["long_shot"].kelly_fraction() == pytest.approx(0.04)
    assert by_id["dice_duel_5"].kelly_fraction() == pytest.approx(0.6)
    assert by_id["dice_duel_3"].kelly_fraction() == 0 and by_id["dice_duel_3"].edge < 0
