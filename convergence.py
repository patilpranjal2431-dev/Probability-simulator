"""How fast does a Monte Carlo estimate approach the exact answer?

Theory: the standard error of p_hat is sqrt(p(1-p)/n), so the typical error shrinks like
1/sqrt(n). On a log-log chart of error vs n that is a straight line with slope -1/2.
Quadrupling the number of trials only halves the error.

This module checks the theory empirically:
  * run many independent replicates of the whole experiment,
  * measure |p_hat - exact| at log-spaced sample sizes,
  * fit the slope of log(error) vs log(n)            (expect about -0.5),
  * count how often the 95% interval contains the exact value (expect about 95%).
"""

import random
from dataclasses import dataclass
from math import log10, pi, sqrt

from .montecarlo import Trial, simulate


@dataclass(frozen=True)
class SizeStats:
    """Error statistics at one sample size, across all replicates."""

    n: int
    mean_abs_error: float
    rmse: float
    error_p05: float  # 5th percentile of |error|  (lower edge of the band)
    error_p95: float  # 95th percentile of |error| (upper edge of the band)
    coverage: float  # fraction of replicates whose 95% Wilson interval contained the exact value


@dataclass(frozen=True)
class ConvergenceResult:
    label: str
    exact: float
    seed: int
    replicates: int
    stats: list[SizeStats]
    slope: float  # fitted slope of log10(mean_abs_error) vs log10(n)
    overall_coverage: float  # coverage pooled over all sizes

    def reference_error(self, n: float) -> float:
        """Theoretical mean absolute error at sample size n: sqrt(2/pi) * sqrt(p(1-p)/n).

        (For a normal distribution the mean absolute deviation is sqrt(2/pi) times the
        standard deviation. This is the "1/sqrt(n)" reference line on the chart.)
        """
        return sqrt(2 / pi) * sqrt(self.exact * (1 - self.exact) / n)


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100) of an ascending list."""
    if not sorted_values:
        raise ValueError("no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    position = (len(sorted_values) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log10(y) against log10(x). Points with y <= 0 are skipped."""
    points = [(log10(x), log10(y)) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
    if len(points) < 2:
        raise ValueError("need at least two positive points")
    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    numerator = sum((px - mean_x) * (py - mean_y) for px, py in points)
    denominator = sum((px - mean_x) ** 2 for px, _ in points)
    if denominator == 0:
        raise ValueError("x values must not all be equal")
    return numerator / denominator


def convergence_study(
    trial: Trial,
    exact: float,
    sizes: list[int],
    replicates: int,
    seed: int,
    label: str = "",
) -> ConvergenceResult:
    """Repeat the experiment `replicates` times and summarise the error at each size.

    Each replicate gets its own seed derived from `seed`, so the whole study is
    reproducible and the replicates are independent of each other.
    """
    if not 0 < exact < 1:
        raise ValueError("exact must be strictly between 0 and 1")
    if replicates < 2:
        raise ValueError("need at least 2 replicates")
    checkpoints = sorted(set(sizes))

    master = random.Random(seed)
    errors: list[list[float]] = [[] for _ in checkpoints]
    covered = [0] * len(checkpoints)
    for _ in range(replicates):
        estimates = simulate(trial, checkpoints, master.getrandbits(63))
        for i, est in enumerate(estimates):
            errors[i].append(abs(est.p_hat - exact))
            covered[i] += est.contains(exact)

    stats = []
    for i, n in enumerate(checkpoints):
        ordered = sorted(errors[i])
        stats.append(
            SizeStats(
                n=n,
                mean_abs_error=sum(ordered) / replicates,
                rmse=sqrt(sum(e * e for e in ordered) / replicates),
                error_p05=percentile(ordered, 5),
                error_p95=percentile(ordered, 95),
                coverage=covered[i] / replicates,
            )
        )

    return ConvergenceResult(
        label=label,
        exact=exact,
        seed=seed,
        replicates=replicates,
        stats=stats,
        slope=loglog_slope([s.n for s in stats], [s.mean_abs_error for s in stats]),
        overall_coverage=sum(covered) / (replicates * len(checkpoints)),
    )
