"""The behaviour EVERY grade store must have (MemoryGradeStore and SqlGradeStore alike).

Each check gets (store, users, new_game): three user ids and a function returning a fresh match id."""

import pytest

from app.grading.stats import aggregate
from app.grading.types import GradedDecision


def row(hand=1, index=0, ev=2.5, regret=0.5, net=10, counts=True, kind="bet", taken=("check", 0), best=("bet", 5)):
    return GradedDecision(
        hand_number=hand, decision_index=index, seat=0, decision_kind=kind, own_value=4,
        action_kind=taken[0], action_amount=taken[1], best_kind=best[0], best_amount=best[1],
        ev_taken=ev, ev_best=ev + regret, regret=regret, hand_net=net, counts_for_ev=counts,
    )


def check_an_empty_store_is_empty(store, users, new_game):
    assert store.decisions_for(users[0]) == [] and store.skill_totals() == {}


def check_decisions_come_back_newest_first_with_every_field(store, users, new_game):
    g1, g2 = new_game(), new_game()
    store.save(g1, users[0], [row(hand=1, net=10), row(hand=2, net=-10)])
    store.save(g2, users[0], [row(hand=1, ev=7.25, regret=0.0, counts=False, kind="respond", taken=("call", 0), best=("call", 0))])
    got = store.decisions_for(users[0])
    assert [(r["game_id"], r["hand_number"]) for r in got] == [(g2, 1), (g1, 2), (g1, 1)]
    assert got[0]["ev_taken"] == pytest.approx(7.25) and got[0]["counts_for_ev"] is False
    assert got[0]["decision_kind"] == "respond" and got[0]["action_kind"] == "call"
    assert got[1]["hand_net"] == -10 and got[1]["best_kind"] == "bet" and got[1]["best_amount"] == 5
    assert set(got[0]) == {"game_id", "hand_number", "decision_index", "seat", "decision_kind", "own_value",
                           "action_kind", "action_amount", "best_kind", "best_amount", "ev_taken", "ev_best",
                           "regret", "hand_net", "counts_for_ev"}


def check_paging_works(store, users, new_game):
    g = new_game()
    store.save(g, users[0], [row(hand=h) for h in range(1, 6)])
    assert [r["hand_number"] for r in store.decisions_for(users[0], limit=2)] == [5, 4]
    assert [r["hand_number"] for r in store.decisions_for(users[0], limit=2, offset=2)] == [3, 2]
    assert [r["hand_number"] for r in store.decisions_for(users[0], limit=10, offset=4)] == [1]
    assert store.decisions_for(users[0], limit=5, offset=5) == []


def check_saving_a_match_again_replaces_it_but_other_matches_are_kept(store, users, new_game):
    g1, g2 = new_game(), new_game()
    store.save(g1, users[0], [row(hand=1), row(hand=2)])
    store.save(g1, users[0], [row(hand=1), row(hand=2)])  # a retry
    assert len(store.decisions_for(users[0])) == 2
    store.save(g1, users[0], [row(hand=1)])  # a corrected version replaces the old one
    assert len(store.decisions_for(users[0])) == 1
    store.save(g2, users[0], [row(hand=1)])
    assert len(store.decisions_for(users[0])) == 2
    assert store.skill_totals()[users[0]].decisions == 2


def check_users_are_kept_apart(store, users, new_game):
    g = new_game()
    store.save(g, users[0], [row(ev=1.0, net=5)])
    store.save(g, users[1], [row(ev=9.0, net=-5), row(hand=2, ev=9.0, net=-5)])
    assert len(store.decisions_for(users[0])) == 1 and len(store.decisions_for(users[1])) == 2
    assert store.decisions_for(users[2]) == []
    totals = store.skill_totals()
    assert totals[users[0]].ev_earned == pytest.approx(1.0) and totals[users[1]].ev_earned == pytest.approx(18.0)
    assert users[2] not in totals


def check_totals_match_the_pure_aggregation(store, users, new_game):
    g1, g2 = new_game(), new_game()
    rows1 = [row(hand=1, index=0, ev=3.0, regret=1.0, net=20, counts=False), row(hand=1, index=1, ev=4.5, regret=0.25, net=20),
             row(hand=2, ev=-1.5, regret=0.0, net=-10)]
    rows2 = [row(hand=1, ev=0.75, regret=2.0, net=-30)]
    store.save(g1, users[0], rows1)
    store.save(g2, users[0], rows2)
    store.save(g1, users[1], [row(ev=6.0, regret=0.0, net=6)])
    expected = aggregate([(users[0], r) for r in rows1 + rows2] + [(users[1], row(ev=6.0, regret=0.0, net=6))])
    actual = store.skill_totals()
    assert set(actual) == set(expected)
    for uid in expected:
        e, a = expected[uid], actual[uid]
        assert (a.hands, a.decisions, a.actual_net) == (e.hands, e.decisions, e.actual_net)
        assert a.ev_earned == pytest.approx(e.ev_earned) and a.regret == pytest.approx(e.regret)
        assert a.ev_potential == pytest.approx(e.ev_potential)


def check_saving_nothing_is_allowed(store, users, new_game):
    store.save(new_game(), users[0], [])
    assert store.decisions_for(users[0]) == [] and users[0] not in store.skill_totals()


ALL_CHECKS = [
    check_an_empty_store_is_empty,
    check_decisions_come_back_newest_first_with_every_field,
    check_paging_works,
    check_saving_a_match_again_replaces_it_but_other_matches_are_kept,
    check_users_are_kept_apart,
    check_totals_match_the_pure_aggregation,
    check_saving_nothing_is_allowed,
]
