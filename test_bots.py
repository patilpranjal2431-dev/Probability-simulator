"""The four bots: legality, and the exact behaviour each one is documented to have."""

import math
import random
from itertools import product

import pytest

from app.games.base import CALL, CHECK, FOLD, bet
from app.games.bots import (
    BOT_NAMES, CautiousBot, EVBot, KellyBot, RandomBot, edge, expected_log_wealth, make_bot,
)
from app.games.dice_duel import DiceDuel
from app.games.kuhn import Kuhn
from app.games.match import play_match
from helpers_games import legal_actions, walk

RNG = random.Random(0)


def dice_decision(roll, stacks=(1000, 1000)):
    """The decision seat 0 faces when opening a hand holding `roll`."""
    return DiceDuel().new_hand(RNG, stacks, 0, deal=(roll, 1)).decision()


def dice_response(roll, bet_size, stacks=(1000, 1000)):
    """The decision seat 1 faces when it holds `roll` and seat 0 has bet `bet_size`."""
    hand = DiceDuel().new_hand(RNG, stacks, 0, deal=(1, roll))
    hand.act(bet(bet_size))
    return hand.decision()


# ------------------------------------------------------------------- registry


def test_make_bot_builds_each_named_bot():
    assert [make_bot(n).name for n in BOT_NAMES] == ["random", "cautious", "ev", "kelly"]
    with pytest.raises(ValueError):
        make_bot("genius")


def test_kelly_rejects_bad_fractions():
    for bad in (0, -0.5, 1.5):
        with pytest.raises(ValueError):
            KellyBot(bad)


# ----------------------------------------------------------------- legality


@pytest.mark.parametrize("name", BOT_NAMES)
@pytest.mark.parametrize("game", [DiceDuel(), Kuhn()], ids=["dice", "kuhn"])
def test_every_bot_always_acts_legally_across_every_state(name, game):
    """Walk EVERY reachable decision (small stacks) and check the bot's action is legal there."""
    bot, rng = make_bot(name), random.Random(5)
    stacks_to_try = [(game.ante, game.ante), (game.ante + 3, game.ante + 40), (500, 500)]
    checked, kinds = 0, set()
    for stacks, first, deal in product(stacks_to_try, (0, 1), game.all_deals()):
        stacks = tuple(max(s, game.ante) for s in stacks)
        # every decision point reachable by ANY play, not just the bot's own
        seen = set()
        for history, _ in walk(game, deal, stacks, first) if sum(stacks) < 200 else [((), None)]:
            for cut in range(len(history) + 1):
                prefix = history[:cut]
                if prefix in seen:
                    continue
                seen.add(prefix)
                hand = game.new_hand(RNG, stacks, first, deal=deal)
                for a in prefix:
                    hand.act(a)
                if hand.done:
                    continue
                d = hand.decision()
                kinds.add(d.kind)
                for _ in range(5):
                    assert d.is_legal(bot.act(d, rng)), (name, d)
                checked += 1
    assert kinds == {"bet", "respond"}  # both kinds of decision were exercised
    assert checked >= 80


@pytest.mark.parametrize("a", BOT_NAMES)
@pytest.mark.parametrize("b", BOT_NAMES)
def test_every_pair_can_play_full_matches_and_chips_are_conserved(a, b):
    for game, stack in ((DiceDuel(), 1000), (DiceDuel(), 30), (Kuhn(), 100), (Kuhn(), 3)):
        for seed in range(4):
            r = play_match(game, (make_bot(a), make_bot(b)), seed, stack, max_hands=60)
            assert sum(r.final_stacks) == 2 * stack and min(r.final_stacks) >= 0


# --------------------------------------------------------------------- random


def test_random_bot_uses_every_option_and_stays_in_range():
    bot, rng = RandomBot(), random.Random(1)
    d = dice_decision(4)
    actions = [bot.act(d, rng) for _ in range(3000)]
    assert CHECK in actions
    sizes = [a.amount for a in actions if a.kind == "bet"]
    assert min(sizes) >= 1 and max(sizes) <= d.max_bet and len(set(sizes)) > 100
    replies = {bot.act(dice_response(3, 50), rng) for _ in range(200)}
    assert replies == {CALL, FOLD}


def test_random_bot_is_reproducible_from_its_rng():
    d = dice_decision(4)
    assert [RandomBot().act(d, random.Random(7)) for _ in range(5)] == [
        RandomBot().act(d, random.Random(7)) for _ in range(5)
    ]


# ------------------------------------------------------------------- cautious


@pytest.mark.parametrize(("roll", "strong"), [(1, False), (2, False), (3, False), (4, False), (5, True), (6, True)])
def test_cautious_bets_small_only_with_strong_hands(roll, strong):
    action = CautiousBot().act(dice_decision(roll), RNG)
    assert action == (bet(49) if strong else CHECK)  # 5% of the 990 behind


def test_cautious_bet_is_at_least_the_minimum():
    assert CautiousBot().act(dice_decision(6, stacks=(20, 20)), RNG) == bet(1)  # 10 behind -> 0 -> 1


@pytest.mark.parametrize(("roll", "calls"), [(3, False), (4, False), (5, True), (6, True)])
def test_cautious_calls_only_with_strong_hands(roll, calls):
    assert CautiousBot().act(dice_response(roll, 100), RNG) == (CALL if calls else FOLD)


# ------------------------------------------------------------------------- ev


@pytest.mark.parametrize(("roll", "expected"), [(1, CHECK), (2, CHECK), (3, CHECK), (4, bet(990)), (5, bet(990)), (6, bet(990))])
def test_ev_bot_goes_all_in_on_any_positive_edge(roll, expected):
    assert EVBot().act(dice_decision(roll), RNG) == expected


def test_edge_formula():
    assert edge(dice_decision(4)) == pytest.approx(1 / 6)
    assert edge(dice_decision(3)) == pytest.approx(-1 / 6)


def test_ev_bot_calls_exactly_when_calling_beats_folding():
    for roll in range(1, 7):
        for size in (1, 10, 100, 990):
            d = dice_response(roll, size)
            call_ev = edge(d) * d.opp_invested
            fold_ev = -d.my_invested
            assert EVBot().act(d, RNG) == (CALL if call_ev > fold_ev else FOLD), (roll, size)
    assert EVBot().act(dice_response(1, 1), RNG) == CALL  # only 1 more coin to risk
    assert EVBot().act(dice_response(3, 100), RNG) == FOLD


# ---------------------------------------------------------------------- kelly


@pytest.mark.parametrize(("roll", "expected"), [(1, CHECK), (2, CHECK), (3, CHECK), (4, bet(90)), (5, bet(290)), (6, bet(490))])
def test_half_kelly_sizes_its_bet_to_its_edge(roll, expected):
    """f* = (w - l)/(w + l): 0.2, 0.6, 1.0 for a 4, 5, 6. Half of that, of the 1000 bankroll,
    minus the 10 already in the pot."""
    assert KellyBot(0.5).act(dice_decision(roll), RNG) == expected


def test_kelly_bets_grow_with_the_edge_and_stay_below_the_ev_bot():
    sizes = [KellyBot().act(dice_decision(r), RNG).amount for r in (4, 5, 6)]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    assert all(s < EVBot().act(dice_decision(r), RNG).amount for s, r in zip(sizes, (4, 5, 6)))


@pytest.mark.parametrize("roll", [4, 5, 6])
def test_full_kelly_matches_a_brute_force_search_of_expected_log_wealth(roll):
    """The closed form should pick the bet that maximises E[ln wealth] over every integer size."""
    d = dice_decision(roll)
    best = max(range(0, d.max_bet + 1), key=lambda b: expected_log_wealth(d, d.my_invested + b))
    chosen = KellyBot(1.0).act(d, RNG)
    size = 0 if chosen == CHECK else chosen.amount
    assert abs(size - best) <= 1


def test_kelly_scales_with_the_bankroll():
    small = KellyBot().act(dice_decision(5, stacks=(200, 200)), RNG).amount
    large = KellyBot().act(dice_decision(5, stacks=(2000, 2000)), RNG).amount
    assert large > 5 * small


def test_kelly_call_decision_compares_expected_log_wealth():
    for roll in range(1, 7):
        for size in (1, 10, 100, 500, 990):
            d = dice_response(roll, size)
            call_value = expected_log_wealth(d, d.opp_invested)
            fold_value = math.log(d.stack)
            assert KellyBot().act(d, RNG) == (CALL if call_value >= fold_value else FOLD)


def test_kelly_never_risks_ruin_on_a_losing_all_in():
    assert KellyBot().act(dice_response(1, 990), RNG) == FOLD
    assert KellyBot().act(dice_response(4, 990), RNG) == FOLD  # 1/6 ruin chance is not worth +EV


def test_expected_log_wealth_is_minus_infinity_when_ruin_is_possible():
    d = dice_response(2, 990)  # calling risks everything
    assert expected_log_wealth(d, d.opp_invested) == -math.inf


def test_in_a_fixed_bet_game_kelly_has_nothing_to_size_and_plays_like_the_ev_bot():
    """Kuhn has one bet size, so Kelly's sizing is irrelevant. Both act identically everywhere."""
    game = Kuhn()
    compared = 0
    for deal, first in product(game.all_deals(), (0, 1)):
        for history, _ in walk(game, deal, (100, 100), first):
            for cut in range(len(history) + 1):
                hand = game.new_hand(RNG, (100, 100), first, deal=deal)
                for a in history[:cut]:
                    hand.act(a)
                if not hand.done:
                    d = hand.decision()
                    assert KellyBot().act(d, RNG) == EVBot().act(d, RNG)
                    compared += 1
    assert compared > 50
