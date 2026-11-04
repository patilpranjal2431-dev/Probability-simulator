"""Grade the decisions of a finished match."""

from app.games.base import CALL, CHECK, FOLD, Action, HandResult, bet
from app.games.registry import make_game

from .evaluate import _opponent_deals, _replay, q_values
from .reference import reference_for
from .types import DecisionContext, GradedDecision

TOLERANCE = 1e-9
SIMPLEST_FIRST = {"check": 0, "call": 0, "fold": 1, "bet": 2}  # when actions tie, report the simplest
ENUMERATE_ALL_BELOW = 64  # with this few legal bet sizes, check every one


def candidate_actions(game, ctx: DecisionContext, reference) -> tuple[str, list[Action]]:
    """(decision kind, the actions worth evaluating). Always includes the action taken."""
    deal = _opponent_deals(game, ctx.seat, ctx.own_value)[0]
    hand, _ = _replay(game, ctx, deal, reference)
    decision = hand.decision()
    taken = Action(ctx.action_kind, ctx.action_amount)
    if decision.kind == "respond":
        return "respond", [CALL, FOLD]

    actions = [CHECK]
    if decision.can_bet():
        low, high = decision.min_bet, decision.max_bet
        if high - low + 1 <= ENUMERATE_ALL_BELOW:
            sizes = set(range(low, high + 1))
        else:  # expected value is linear between the points where the opponent's reply changes
            sizes = {low, high}
            sizes.update(s for s in reference.bet_breakpoints(hand) if low <= s <= high)
        if taken.kind == "bet":
            sizes.add(taken.amount)
        actions += [bet(size) for size in sorted(sizes)]
    return "bet", actions


def grade_decision(game, ctx: DecisionContext, reference) -> dict:
    """Expected value of the action taken, of the best action, and the regret."""
    kind, candidates = candidate_actions(game, ctx, reference)
    taken = Action(ctx.action_kind, ctx.action_amount)
    values = q_values(game, ctx, reference, candidates)
    ev_taken = values[taken]
    ev_best = max(values.values())
    if ev_taken >= ev_best - TOLERANCE:
        best, regret = taken, 0.0  # the action taken is (one of) the best
    else:
        best = min((a for a, v in values.items() if v >= ev_best - TOLERANCE), key=lambda a: (SIMPLEST_FIRST[a.kind], a.amount))
        regret = ev_best - ev_taken
    return {
        "decision_kind": kind, "best": best,
        "ev_taken": ev_taken, "ev_best": max(ev_best, ev_taken), "regret": regret,
    }


def grade_match(
    game_name: str, log: list[DecisionContext], results: list[HandResult]
) -> dict[int, list[GradedDecision]]:
    """Grade every decision a player made themselves (automatic moves are skipped).

    Returns the graded decisions for each seat. A hand counts towards a player's EV total only
    through their LAST own decision, and only if the table never played for them in that hand,
    so nothing is counted twice and nobody is credited for moves they did not make.
    """
    game = make_game(game_name)
    reference = reference_for(game.name)

    had_auto = {(c.hand_number, c.seat) for c in log if c.auto}
    own = [c for c in log if not c.auto]
    last_index = {}
    for c in own:
        key = (c.hand_number, c.seat)
        last_index[key] = max(last_index.get(key, -1), c.decision_index)

    graded: dict[int, list[GradedDecision]] = {0: [], 1: []}
    for c in own:
        g = grade_decision(game, c, reference)
        key = (c.hand_number, c.seat)
        graded[c.seat].append(
            GradedDecision(
                hand_number=c.hand_number, decision_index=c.decision_index, seat=c.seat,
                decision_kind=g["decision_kind"], own_value=c.own_value,
                action_kind=c.action_kind, action_amount=c.action_amount,
                best_kind=g["best"].kind, best_amount=g["best"].amount,
                ev_taken=g["ev_taken"], ev_best=g["ev_best"], regret=g["regret"],
                hand_net=results[c.hand_number - 1].net[c.seat],
                counts_for_ev=(c.decision_index == last_index[key] and key not in had_auto),
            )
        )
    return graded
