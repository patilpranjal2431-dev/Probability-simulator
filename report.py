"""Run convergence studies and write the charts.

    cd backend
    python -m app.probability.report --out ../docs/charts

Prints a table for each event and writes one SVG per event.
"""

import argparse
from pathlib import Path

from . import exact, montecarlo
from .chart import write_chart
from .convergence import ConvergenceResult, convergence_study


def studies(max_n: int, replicates: int, seed: int) -> list[ConvergenceResult]:
    """The standard set of events, each compared with its exact probability."""
    sizes = montecarlo.log_spaced_sizes(max_n)
    poker_sizes = montecarlo.log_spaced_sizes(max(100, max_n // 2))  # poker trials are slower
    poker_reps = max(20, replicates // 2)
    return [
        convergence_study(
            montecarlo.dice_sum_trial(2, 7),
            float(exact.dice_sum_probability(2, 7)),
            sizes, replicates, seed, "two dice sum to 7"),
        convergence_study(
            montecarlo.birthday_trial(23),
            float(exact.birthday_match_probability(23)),
            sizes, replicates, seed, "birthday match among 23 people"),
        convergence_study(
            montecarlo.hypergeometric_trial(52, 4, 5, 2),
            float(exact.hypergeometric_pmf(52, 4, 5, 2)),
            sizes, replicates, seed, "exactly 2 aces in a 5-card hand"),
        convergence_study(
            montecarlo.poker_trial("one_pair"),
            float(exact.poker_hand_probabilities()["one_pair"]),
            poker_sizes, poker_reps, seed, "one pair in 5-card poker"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="../docs/charts", help="folder for the SVG charts")
    parser.add_argument("--max-n", type=int, default=10_000)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    for result in studies(args.max_n, args.replicates, args.seed):
        print(f"\n{result.label}   exact p = {result.exact:.5f}   "
              f"slope = {result.slope:.3f} (theory -0.5)   coverage = {result.overall_coverage:.1%}")
        print(f"{'n':>8} {'mean |err|':>11} {'theory':>9} {'coverage':>9}")
        for s in result.stats:
            print(f"{s.n:>8} {s.mean_abs_error:>11.5f} {result.reference_error(s.n):>9.5f} {s.coverage:>9.1%}")
        name = "".join(c if c.isalnum() else "_" for c in result.label.lower()).strip("_")
        path = write_chart(result, Path(args.out) / f"{name}.svg")
        print(f"chart written: {path}")


if __name__ == "__main__":
    main()
