"""Whole matches and exact per-hand analysis."""

import random
from fractions import Fraction

import pytest

from app.games.analysis import exact_hand_ev
from app.games.base import CALL, CHECK, bet
from app.games.bots import make_bot
from app.games.dice_duel import DiceDuel
from app.games.kuhn import Kuhn
from app.games.match import MatchResult, play_match


class AlwaysCheckCall:
    """Test bot: checks when it can, calls when asked. Deterministic."""

    name = "check_call"

    def act(self, decision, rng):
        return CHECK if decision.kind == "bet" else CALL


# ---------------------------------------------------------------------- match


def test_chips_are_conserved_and_never_negative():
    for seed in range(20):
        r = play_match(DiceDuel(), (make_bot("ev"), make_bot("random")), seed)
        assert sum(r.final_stacks) == 2 * r.start_stack and min(r.final_stacks) >= 0
        assert r.net(0) + r.net(1) == 0


def test_same_seed_replays_exactly_and_different_seeds_differ():
    bots = (make_bot("random"), make_bot("kelly"))
    assert play_match(DiceDuel(), bots, 5) == play_match(DiceDuel(), bots, 5)
    # many matches end in the same bust score, so compare length as well
    results = {(m.hands, m.final_stacks) for m in (play_match(DiceDuel(), bots, s) for s in range(10))}
    assert len(results) >= 6


def test_match_never_exceeds_max_hands_and_reports_why_it_ended():
    r = play_match(Kuhn(), (make_bot("cautious"), make_bot("cautious")), 1, max_hands=10)
    assert r.hands == 10 and r.ended_by == "max_hands"
    busted = play_match(DiceDuel(), (make_bot("ev"), make_bot("ev")), 3, starting_stack=30, max_hands=500)
    assert busted.ended_by == "bust" and busted.hands < 500
    assert min(busted.final_stacks) < DiceDuel().ante


def test_winner_is_the_seat_with_more_chips():
    assert MatchResult(10, 100, (120, 80), "max_hands").winner == 0
    assert MatchResult(10, 100, (80, 120), "max_hands").winner == 1
    assert MatchResult(10, 100, (100, 100), "max_hands").winner is None


def test_check_call_bots_play_the_full_distance_and_keep_their_chips_close():
    r = play_match(DiceDuel(), (AlwaysCheckCall(), AlwaysCheckCall()), 4, max_hands=50)
    assert r.hands == 50 and r.ended_by == "max_hands"
    assert max(abs(r.net(0)), abs(r.net(1))) <= 50 * 10  # at most the ante per hand


def test_match_rejects_bad_arguments():
    bots = (make_bot("ev"), make_bot("ev"))
    with pytest.raises(ValueError):
        play_match(DiceDuel(), bots, 1, starting_stack=5)  # cannot cover the ante
    with pytest.raises(ValueError):
        play_match(DiceDuel(), bots, 1, max_hands=0)


# -------------------------------------------------------------- exact analysis


def test_two_check_call_bots_have_exactly_zero_expected_value():
    for game in (DiceDuel(), Kuhn()):
        for first in (0, 1):
            assert exact_hand_ev(game, (AlwaysCheckCall(), AlwaysCheckCall()), first) == 0


def test_exact_ev_matches_a_hand_calculation():
    """Cautious (acts first) vs a bot that always calls, worked out by hand.

    Roll 5 or 6 -> bets 49 and is always called; otherwise checks for the 10-chip antes.
    Edge for a roll r vs a uniform opponent: (2r - 7)/6.
    """
    expected = Fraction(0)
    for roll in range(1, 7):
        edge = Fraction(2 * roll - 7, 6)
        stake = 10 + 49 if roll >= 5 else 10
        expected += edge * stake / 6
    assert exact_hand_ev(DiceDuel(), (make_bot("cautious"), AlwaysCheckCall()), first=0) == expected


def test_swapping_seats_negates_the_exact_value():
    for game in (DiceDuel(), Kuhn()):
        for a, b in (("cautious", "ev"), ("ev", "kelly"), ("kelly", "cautious")):
            one = exact_hand_ev(game, (make_bot(a), make_bot(b)), first=0)
            swapped = exact_hand_ev(game, (make_bot(b), make_bot(a)), first=1)
            assert one == -swapped


def test_a_bot_playing_itself_is_fair_once_seats_are_averaged():
    for game in (DiceDuel(), Kuhn()):
        for name in ("cautious", "ev", "kelly"):
            bots = (make_bot(name), make_bot(name))
            assert exact_hand_ev(game, bots, 0) + exact_hand_ev(game, bots, 1) == 0


def test_the_exact_numbers_that_explain_the_tournament_results():
    """Per-hand exact EVs (averaged over who acts first, full stacks) behind the story:
    in Dice Duel the risk-neutral EV bot is crushed by Kelly, because it goes all-in on any edge."""
    game = DiceDuel()

    def avg(a, b):
        bots = (make_bot(a), make_bot(b))
        return (exact_hand_ev(game, bots, 0) + exact_hand_ev(game, bots, 1)) / 2

    assert avg("kelly", "ev") > 30
    assert avg("cautious", "ev") > 25
    assert avg("kelly", "kelly") == 0

    kuhn = Kuhn()
    bots = (make_bot("cautious"), make_bot("ev"))
    assert (exact_hand_ev(kuhn, bots, 0) + exact_hand_ev(kuhn, bots, 1)) / 2 == Fraction(1, 6)


def test_random_bots_have_no_exact_value():
    with pytest.raises(RuntimeError):
        exact_hand_ev(DiceDuel(), (make_bot("random"), make_bot("ev")), 0)
