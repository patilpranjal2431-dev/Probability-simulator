"""Monte Carlo simulation of a bankroll over many paths.

Definitions
  * RUIN: the bankroll ever touched `ruin_fraction` x its starting value (0 means "went broke").
    Paths keep playing after touching it, so you can still see where they end up.
  * TARGET (optional): a path stops once it reaches this bankroll (a profit goal).
  * DRAWDOWN: the biggest fall from a previous peak, as a fraction of that peak.
  * A path that reaches exactly 0 cannot bet any more and stays there.

Randomness is a SEEDED generator, so a simulation can be reproduced exactly from its inputs.
"""

import random
from dataclasses import dataclass
from math import isfinite, log

from app.probability.convergence import percentile
from app.probability.montecarlo import wilson_interval

from .model import BetSpec

QUANTILES = (5, 25, 50, 75, 95)
MAX_PATHS, MAX_BETS, MAX_STEPS = 20_000, 5_000, 4_000_000


@dataclass(frozen=True)
class Strategy:
    """How much to stake each time.

    kind "kelly":    stake `value` x the Kelly fraction of the CURRENT bankroll (1 = full Kelly, 0.5 = half)
    kind "fraction": stake the fixed fraction `value` of the current bankroll
    kind "flat":     stake the same `value` coins every time (never more than you have)
    """

    kind: str
    value: float

    def __post_init__(self) -> None:
        if self.kind not in ("kelly", "fraction", "flat"):
            raise ValueError("strategy kind must be kelly, fraction or flat")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not self.value > 0:
            raise ValueError("strategy value must be a positive number")
        if self.kind == "kelly" and self.value > 5:
            raise ValueError("a Kelly multiple above 5 is not supported")
        if self.kind == "fraction" and self.value > 1:
            raise ValueError("a fraction of the bankroll cannot exceed 1")

    def stake_fraction(self, spec: BetSpec) -> float | None:
        """The fraction of the bankroll staked each bet (None for flat staking)."""
        if self.kind == "flat":
            return None
        if self.kind == "fraction":
            return float(self.value)
        return min(1.0, self.value * spec.kelly_fraction())


@dataclass(frozen=True)
class BankrollResult:
    spec: BetSpec
    strategy: Strategy
    bets: int
    paths: int
    start: float
    ruin_fraction: float
    target: float | None
    seed: int
    stake_fraction: float | None
    kelly_fraction: float
    # outcomes
    ruin_probability: float
    ruin_ci: tuple[float, float]
    median_time_to_ruin: float | None
    target_probability: float | None
    prob_below_start: float
    prob_double: float
    fraction_wiped_out: float
    mean_final: float
    median_final: float
    final_quantiles: dict[int, float]
    median_max_drawdown: float
    p90_max_drawdown: float
    mean_log_growth: float | None  # per bet; None if any path ended at 0
    theory_growth_rate: float | None
    theory_mean_final: float | None
    # for the charts
    finals: list[float]
    times: list[int]
    series: dict[int, list[float]]  # quantile -> bankroll at each time in `times`
    sample_paths: list[list[float]]

    def summary(self) -> dict:
        """The numbers (no big arrays), JSON-friendly."""
        keys = (
            "bets", "paths", "start", "ruin_fraction", "target", "seed", "stake_fraction", "kelly_fraction",
            "ruin_probability", "median_time_to_ruin", "target_probability", "prob_below_start", "prob_double",
            "fraction_wiped_out", "mean_final", "median_final", "median_max_drawdown", "p90_max_drawdown",
            "mean_log_growth", "theory_growth_rate", "theory_mean_final",
        )
        out = {k: finite_or_none(getattr(self, k)) for k in keys}
        out["ruin_ci"] = list(self.ruin_ci)
        out["final_quantiles"] = {str(q): v for q, v in self.final_quantiles.items()}
        out["strategy"] = {"kind": self.strategy.kind, "value": self.strategy.value}
        out["bet"] = {"win": self.spec.win, "lose": self.spec.lose, "push": self.spec.push, "odds": self.spec.odds,
                      "edge": self.spec.edge}
        return out


def finite_or_none(value):
    """JSON cannot carry infinity or NaN (staking everything has a growth rate of minus infinity)."""
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _grid(bets: int, points: int) -> list[int]:
    return sorted({round(i * bets / points) for i in range(points + 1)} | {0, bets})


def simulate_bankroll(
    spec: BetSpec,
    strategy: Strategy,
    bets: int,
    paths: int,
    start: float = 1000.0,
    ruin_fraction: float = 0.1,
    target: float | None = None,
    seed: int = 0,
    sample_paths: int = 8,
    grid_points: int = 60,
) -> BankrollResult:
    if not (isinstance(bets, int) and not isinstance(bets, bool) and 1 <= bets <= MAX_BETS):
        raise ValueError(f"bets must be a whole number from 1 to {MAX_BETS}")
    if not (isinstance(paths, int) and not isinstance(paths, bool) and 1 <= paths <= MAX_PATHS):
        raise ValueError(f"paths must be a whole number from 1 to {MAX_PATHS}")
    if bets * paths > MAX_STEPS:
        raise ValueError(f"bets x paths may not exceed {MAX_STEPS:,}")
    if not start > 0:
        raise ValueError("start must be positive")
    if not 0 <= ruin_fraction < 1:
        raise ValueError("ruin_fraction must be at least 0 and below 1")
    if target is not None and not target > start:
        raise ValueError("target must be above the starting bankroll")

    fraction = strategy.stake_fraction(spec)
    flat_stake = float(strategy.value) if strategy.kind == "flat" else 0.0
    threshold = ruin_fraction * start
    win, win_or_lose, odds = spec.win, spec.win + spec.lose, spec.odds

    rng = random.Random(seed)
    times = _grid(bets, grid_points)
    wanted = set(times)
    columns: dict[int, list[float]] = {t: [] for t in times}
    samples = [[] for _ in range(min(sample_paths, paths))]
    finals, drawdowns, ruin_times = [], [], []
    ruined_count = target_count = 0

    for path in range(paths):
        bankroll = peak = start
        max_drawdown = 0.0
        ruin_time = 0 if bankroll <= threshold else None
        reached = False
        columns[0].append(bankroll)
        if path < len(samples):
            samples[path].append(bankroll)
        for t in range(1, bets + 1):
            if bankroll > 0 and not reached:
                stake = bankroll * fraction if fraction is not None else min(flat_stake, bankroll)
                u = rng.random()
                if u < win:
                    bankroll += odds * stake
                elif u < win_or_lose:
                    bankroll -= stake
                if bankroll > peak:
                    peak = bankroll
                else:
                    drop = (peak - bankroll) / peak
                    if drop > max_drawdown:
                        max_drawdown = drop
                if ruin_time is None and bankroll <= threshold:
                    ruin_time = t
                if target is not None and bankroll >= target:
                    reached = True
            if t in wanted:
                columns[t].append(bankroll)
                if path < len(samples):
                    samples[path].append(bankroll)
        finals.append(bankroll)
        drawdowns.append(max_drawdown)
        if ruin_time is not None:
            ruined_count += 1
            ruin_times.append(ruin_time)
        target_count += reached

    ordered = sorted(finals)
    wiped = sum(1 for v in finals if v <= 1e-12)
    log_growth = None if wiped else sum(log(v / start) for v in finals) / (paths * bets)
    theory_growth = spec.growth_rate(fraction) if fraction is not None else None
    theory_mean = start * (1 + fraction * spec.edge) ** bets if fraction is not None and target is None else None
    dd_sorted = sorted(drawdowns)

    return BankrollResult(
        spec=spec, strategy=strategy, bets=bets, paths=paths, start=start, ruin_fraction=ruin_fraction,
        target=target, seed=seed, stake_fraction=fraction, kelly_fraction=spec.kelly_fraction(),
        ruin_probability=ruined_count / paths, ruin_ci=wilson_interval(ruined_count, paths),
        median_time_to_ruin=percentile(sorted(ruin_times), 50) if ruin_times else None,
        target_probability=target_count / paths if target is not None else None,
        prob_below_start=sum(1 for v in finals if v < start) / paths,
        prob_double=sum(1 for v in finals if v >= 2 * start) / paths,
        fraction_wiped_out=wiped / paths,
        mean_final=sum(finals) / paths, median_final=percentile(ordered, 50),
        final_quantiles={q: percentile(ordered, q) for q in QUANTILES},
        median_max_drawdown=percentile(dd_sorted, 50), p90_max_drawdown=percentile(dd_sorted, 90),
        mean_log_growth=log_growth, theory_growth_rate=theory_growth, theory_mean_final=theory_mean,
        finals=ordered, times=times,
        series={q: [percentile(sorted(columns[t]), q) for t in times] for q in QUANTILES},
        sample_paths=samples,
    )
