"""Bot-vs-bot tournaments with confidence intervals.

    cd backend
    python -m app.games.tournament --game dice --matches 300
    python -m app.games.tournament --game kuhn --matches 300 --mirror

What is measured, for bot A against bot B over many independent matches:
  * win rate           matches in which A finished with MORE chips (a draw is not a win);
                       95% Wilson confidence interval
  * net per match      A's final stack minus its starting stack; 95% normal-approximation interval
  * net per hand       TOTAL net chips / TOTAL hands over all matches (a ratio estimator, so long
                       matches count more than short ones); 95% interval by linearisation
Seats alternate between matches so neither bot always sits first.
"""

import argparse
import random
from dataclasses import dataclass
from itertools import combinations
from math import sqrt

from app.probability.montecarlo import wilson_interval

from .base import Game
from .bots import BOT_NAMES, make_bot
from .dice_duel import DiceDuel
from .kuhn import Kuhn
from .match import play_match

Z_95 = 1.959963984540054


def mean_confidence_interval(values: list[float], z: float = Z_95) -> tuple[float, float, float]:
    """(mean, low, high): normal-approximation interval for the mean (use with 30+ values)."""
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values")
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    half_width = z * sqrt(variance / n)
    return mean, mean - half_width, mean + half_width


def ratio_confidence_interval(
    numerators: list[float], denominators: list[float], z: float = Z_95
) -> tuple[float, float, float]:
    """(ratio, low, high) for  sum(numerators) / sum(denominators)  across independent matches.

    Averaging each match's own rate would let a 10-hand match count as much as a 200-hand one.
    The pooled ratio does not. Its standard error uses the usual linearisation:
        Var(R) ~ sum((y_i - R * h_i)^2) / (n - 1) / (n * mean(h)^2)
    """
    n = len(numerators)
    if n < 2 or n != len(denominators):
        raise ValueError("need at least 2 paired values")
    total_h = sum(denominators)
    if total_h <= 0:
        raise ValueError("denominators must sum to a positive number")
    ratio = sum(numerators) / total_h
    mean_h = total_h / n
    residual_ss = sum((y - ratio * h) ** 2 for y, h in zip(numerators, denominators, strict=True))
    standard_error = sqrt(residual_ss / (n - 1) / n) / mean_h
    return ratio, ratio - z * standard_error, ratio + z * standard_error


@dataclass(frozen=True)
class PairingResult:
    game: str
    bot_a: str
    bot_b: str
    matches: int
    a_wins: int
    b_wins: int
    draws: int
    win_rate: float  # A's wins / matches
    win_rate_ci: tuple[float, float]
    net_per_match: float  # A's average net chips per match
    net_per_match_ci: tuple[float, float]
    net_per_hand: float
    net_per_hand_ci: tuple[float, float]
    avg_hands: float
    bust_rate: float  # fraction of matches that ended because someone ran out of chips


def run_pairing(
    game: Game,
    bot_a: str,
    bot_b: str,
    matches: int,
    seed: int,
    starting_stack: int | None = None,
    max_hands: int = 200,
) -> PairingResult:
    """Play `matches` independent matches of A vs B and summarise A's results."""
    if matches < 2:
        raise ValueError("need at least 2 matches")
    master = random.Random(seed)
    a_wins = b_wins = draws = busts = total_hands = 0
    nets: list[float] = []
    hand_counts: list[float] = []

    for index in range(matches):
        a_seat = index % 2  # alternate seats between matches
        bots = (make_bot(bot_a), make_bot(bot_b)) if a_seat == 0 else (make_bot(bot_b), make_bot(bot_a))
        result = play_match(game, bots, master.getrandbits(63), starting_stack, max_hands)
        net = result.net(a_seat)
        nets.append(net)
        hand_counts.append(result.hands)
        total_hands += result.hands
        busts += result.ended_by == "bust"
        if result.winner is None:
            draws += 1
        elif result.winner == a_seat:
            a_wins += 1
        else:
            b_wins += 1

    net_mean, net_lo, net_hi = mean_confidence_interval(nets)
    rate_mean, rate_lo, rate_hi = ratio_confidence_interval(nets, hand_counts)
    return PairingResult(
        game=game.name, bot_a=bot_a, bot_b=bot_b, matches=matches,
        a_wins=a_wins, b_wins=b_wins, draws=draws,
        win_rate=a_wins / matches, win_rate_ci=wilson_interval(a_wins, matches),
        net_per_match=net_mean, net_per_match_ci=(net_lo, net_hi),
        net_per_hand=rate_mean, net_per_hand_ci=(rate_lo, rate_hi),
        avg_hands=total_hands / matches, bust_rate=busts / matches,
    )


def round_robin(
    game: Game,
    bot_names: tuple[str, ...] = BOT_NAMES,
    matches: int = 300,
    seed: int = 2026,
    starting_stack: int | None = None,
    max_hands: int = 200,
    mirror: bool = False,
) -> list[PairingResult]:
    """Every pair of bots plays; with `mirror` each bot also plays a copy of itself (sanity check)."""
    pairs = list(combinations(bot_names, 2))
    if mirror:
        pairs += [(name, name) for name in bot_names]
    master = random.Random(seed)
    return [
        run_pairing(game, a, b, matches, master.getrandbits(63), starting_stack, max_hands)
        for a, b in pairs
    ]


def format_results(results: list[PairingResult]) -> str:
    """A readable table (also valid Markdown)."""
    header = (
        "| A vs B | A win rate [95% CI] | A net / match [95% CI] | A net / hand [95% CI] "
        "| draws | busts | hands |\n|---|---|---|---|---|---|---|"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r.bot_a} vs {r.bot_b} "
            f"| {r.win_rate:.1%} [{r.win_rate_ci[0]:.1%}, {r.win_rate_ci[1]:.1%}] "
            f"| {r.net_per_match:+.1f} [{r.net_per_match_ci[0]:+.1f}, {r.net_per_match_ci[1]:+.1f}] "
            f"| {r.net_per_hand:+.2f} [{r.net_per_hand_ci[0]:+.2f}, {r.net_per_hand_ci[1]:+.2f}] "
            f"| {r.draws} | {r.bust_rate:.0%} | {r.avg_hands:.0f} |"
        )
    return header + "\n" + "\n".join(rows)


GAMES = {"dice": DiceDuel, "kuhn": Kuhn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=sorted(GAMES), default="dice")
    parser.add_argument("--matches", type=int, default=300, help="matches per pairing")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument("--mirror", action="store_true", help="also play each bot against itself")
    parser.add_argument("--bots", default=",".join(BOT_NAMES),
                        help="comma-separated bots; in each pair the earlier one is 'A'")
    args = parser.parse_args()
    bot_names = tuple(name.strip() for name in args.bots.split(","))
    for name in bot_names:
        make_bot(name)  # fail early with a clear message on a typo

    game = GAMES[args.game]()
    print(f"{game.name}: ante {game.ante}, starting stack {game.default_stack}, "
          f"up to {args.max_hands} hands, {args.matches} matches per pairing, seed {args.seed}\n")
    print(format_results(round_robin(game, bot_names, matches=args.matches, seed=args.seed,
                                     max_hands=args.max_hands, mirror=args.mirror)))


if __name__ == "__main__":
    main()
