"""Per-player totals and the two leaderboards. Pure functions: no database, no web.

SKILL leaderboard   ranked by cumulative EV earned: what your decisions were WORTH on average,
                    ignoring how the dice and the opponent's cards actually fell out.
COINS leaderboard   ranked by net coins won: what actually happened (luck + skill).

Ranks are sequential (1, 2, 3, ...); exact ties are broken by lower regret, then by user id,
so the order is always deterministic.
"""

from collections.abc import Iterable

from .types import GradedDecision, SkillTotals

Names = dict[int, tuple[str, bool]]  # user id -> (username, is_bot)


def aggregate(rows: Iterable[tuple[int, GradedDecision]]) -> dict[int, SkillTotals]:
    """Sum graded decisions per user. `rows` are (user_id, decision) pairs."""
    acc: dict[int, list] = {}
    for user_id, d in rows:
        a = acc.setdefault(user_id, [0, 0, 0.0, 0.0, 0, 0.0])  # hands, decisions, ev, regret, actual, potential
        a[1] += 1
        a[3] += d.regret
        if d.counts_for_ev:
            a[0] += 1
            a[2] += d.ev_taken
            a[4] += d.hand_net
            a[5] += d.ev_best
    return {
        user_id: SkillTotals(user_id, hands, decisions, ev, regret, actual, potential)
        for user_id, (hands, decisions, ev, regret, actual, potential) in acc.items()
    }


def skill_ranking(totals: dict[int, SkillTotals], names: Names, min_hands: int = 1) -> list[dict]:
    """The skill leaderboard: humans with at least `min_hands` counted hands, best EV first."""
    eligible = [t for t in totals.values() if t.hands >= min_hands and not names.get(t.user_id, ("", True))[1]]
    eligible.sort(key=lambda t: (-t.ev_earned, t.regret, t.user_id))
    return [
        {
            "rank": position,
            "user_id": t.user_id,
            "username": names[t.user_id][0],
            "hands": t.hands,
            "decisions": t.decisions,
            "ev_earned": t.ev_earned,
            "ev_per_hand": t.ev_earned / t.hands,
            "ev_potential": t.ev_potential,
            "regret": t.regret,
            "regret_per_decision": t.regret / t.decisions if t.decisions else 0.0,
            "actual_net": t.actual_net,
            "luck": t.luck,
        }
        for position, t in enumerate(eligible, start=1)
    ]


def coin_ranking(net_coins: dict[int, int], names: Names) -> list[dict]:
    """The coins leaderboard: humans ranked by net coins won across all matches."""
    eligible = [(uid, net) for uid, net in net_coins.items() if not names.get(uid, ("", True))[1]]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [
        {"rank": position, "user_id": uid, "username": names[uid][0], "net_coins": net}
        for position, (uid, net) in enumerate(eligible, start=1)
    ]


def with_rank_gap(skill_rows: list[dict], coin_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Add each player's rank on the OTHER board, so the luck gap is visible.

    rank_gap = coin rank - skill rank. Positive: the player ranks better on skill than on coins
    (their results were worse than their decisions deserved). Negative: they finished higher on
    coins than their decisions earned (they ran well). Zero: the two boards agree."""
    skill_rank = {row["user_id"]: row["rank"] for row in skill_rows}
    coin_rank = {row["user_id"]: row["rank"] for row in coin_rows}
    for row in skill_rows:
        row["coin_rank"] = coin_rank.get(row["user_id"])
        row["rank_gap"] = None if row["coin_rank"] is None else row["coin_rank"] - row["rank"]
    for row in coin_rows:
        row["skill_rank"] = skill_rank.get(row["user_id"])
        row["rank_gap"] = None if row["skill_rank"] is None else row["rank"] - row["skill_rank"]
    return skill_rows, coin_rows
