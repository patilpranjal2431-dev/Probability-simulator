"""Totals and the two leaderboards."""

import pytest

from app.grading.stats import aggregate, coin_ranking, skill_ranking, with_rank_gap
from app.grading.types import GradedDecision, SkillTotals


def gd(ev=1.0, regret=0.0, net=0, counts=True, hand=1, index=0):
    return GradedDecision(hand, index, 0, "bet", 3, "check", 0, "check", 0, ev, ev + regret, regret, net, counts)


NAMES = {1: ("alice", False), 2: ("bob", False), 3: ("carol", False), 9: ("bot_kelly", True)}


def test_aggregate_sums_ev_only_over_counted_hands_but_regret_over_all_decisions():
    rows = [
        (1, gd(ev=5.0, regret=1.0, net=10, counts=False, index=0)),  # an earlier decision in a hand
        (1, gd(ev=8.0, regret=0.5, net=10, counts=True, index=1)),  # the hand's last decision
        (1, gd(ev=-2.0, regret=0.0, net=-10, counts=True, hand=2)),
        (2, gd(ev=3.0, regret=4.0, net=0, counts=True)),
    ]
    totals = aggregate(rows)
    a = totals[1]
    assert (a.hands, a.decisions) == (2, 3)
    assert a.ev_earned == pytest.approx(6.0) and a.regret == pytest.approx(1.5) and a.actual_net == 0
    assert a.luck == pytest.approx(-6.0)  # actual 0 minus expected 6
    assert totals[2] == SkillTotals(2, 1, 1, 3.0, 4.0, 0, 7.0)  # potential = ev_taken + regret = 3 + 4
    # deal value = what perfect play would earn; EV earned is that minus the regret in the counted hands
    assert a.ev_potential == pytest.approx(8.5 + (-2.0 + 0.0))


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {}


def test_skill_ranking_orders_by_ev_then_regret_then_id():
    totals = {
        1: SkillTotals(1, 10, 12, 50.0, 9.0, 40),
        2: SkillTotals(2, 10, 12, 80.0, 30.0, 90),
        3: SkillTotals(3, 10, 12, 50.0, 2.0, 10),  # same EV as alice, less regret -> ahead of her
    }
    rows = skill_ranking(totals, NAMES)
    assert [r["username"] for r in rows] == ["bob", "carol", "alice"]
    assert [r["rank"] for r in rows] == [1, 2, 3]


def test_skill_ranking_derived_columns():
    row = skill_ranking({1: SkillTotals(1, 4, 8, 20.0, 6.0, 30, 26.0)}, NAMES)[0]
    assert row["ev_per_hand"] == 5.0 and row["regret_per_decision"] == 0.75 and row["ev_potential"] == 26.0
    assert row["luck"] == 10.0 and row["actual_net"] == 30


def test_bots_and_unknown_users_never_appear_and_min_hands_is_enforced():
    totals = {1: SkillTotals(1, 3, 4, 9.0, 1.0, 0), 9: SkillTotals(9, 50, 60, 999.0, 0.0, 0),
              2: SkillTotals(2, 1, 1, 100.0, 0.0, 0), 77: SkillTotals(77, 5, 5, 5.0, 0.0, 0)}
    assert [r["username"] for r in skill_ranking(totals, NAMES, min_hands=2)] == ["alice"]
    assert [r["username"] for r in skill_ranking(totals, NAMES, min_hands=1)] == ["bob", "alice"]


def test_coin_ranking_orders_by_net_coins_and_excludes_bots():
    rows = coin_ranking({1: -120, 2: 300, 3: 300, 9: 5000}, NAMES)
    assert [(r["rank"], r["username"], r["net_coins"]) for r in rows] == [
        (1, "bob", 300), (2, "carol", 300), (3, "alice", -120)]


def test_rank_gap_shows_who_was_lucky_and_who_was_not():
    skill = skill_ranking({
        1: SkillTotals(1, 10, 10, 100.0, 1.0, -50),  # best decisions, lost coins: unlucky
        2: SkillTotals(2, 10, 10, 10.0, 40.0, 200),  # poor decisions, won coins: lucky
    }, NAMES)
    coins = coin_ranking({1: -50, 2: 200}, NAMES)
    skill, coins = with_rank_gap(skill, coins)
    by_name = {r["username"]: r for r in skill}
    assert by_name["alice"]["rank"] == 1 and by_name["alice"]["coin_rank"] == 2 and by_name["alice"]["rank_gap"] == 1
    assert by_name["bob"]["rank"] == 2 and by_name["bob"]["coin_rank"] == 1 and by_name["bob"]["rank_gap"] == -1
    assert {r["username"]: r["skill_rank"] for r in coins} == {"alice": 1, "bob": 2}


def test_a_player_on_only_one_board_has_no_gap():
    skill, coins = with_rank_gap(skill_ranking({1: SkillTotals(1, 1, 1, 1.0, 0.0, 0)}, NAMES),
                                 coin_ranking({2: 10}, NAMES))
    assert skill[0]["coin_rank"] is None and skill[0]["rank_gap"] is None
    assert coins[0]["skill_rank"] is None and coins[0]["rank_gap"] is None
