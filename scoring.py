"""Brier scores and calibration statistics. Pure functions.

Brier score of one forecast f (a probability) for an event that did (o = 1) or did not (o = 0) happen:
        BS = (f - o)^2          0 is perfect, 1 is the worst possible, always saying 50% scores 0.25
Lower is better.

Here the TRUE probability p of every event is known exactly, which gives three more numbers:
  * expected Brier   = p*(1-f)^2 + (1-p)*f^2       the score you would average over many repeats
  * uncertainty      = p*(1-p)                      the expected score of someone who KNOWS p. Nobody can beat it
  * excess (reliability) = (f - p)^2                what your forecast cost you beyond that unavoidable noise
and the identity   expected Brier = uncertainty + excess.
The realized Brier score is noisy (one coin flip can move it a lot); the excess is not.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt

BASELINE_BRIER = 0.25  # the score of always forecasting 50%, whatever happens


@dataclass(frozen=True)
class Answer:
    kind: str
    forecast: float
    probability: float  # the true probability
    outcome: bool


def _check(forecast: float) -> None:
    if isinstance(forecast, bool) or not isinstance(forecast, (int, float)) or not isfinite(forecast) or not 0 <= forecast <= 1:
        raise ValueError("a forecast must be a number between 0 and 1")


def brier(forecast: float, outcome: bool) -> float:
    _check(forecast)
    return (forecast - (1.0 if outcome else 0.0)) ** 2


def expected_brier(forecast: float, probability: float) -> float:
    _check(forecast)
    return probability * (1 - forecast) ** 2 + (1 - probability) * forecast**2


def excess_brier(forecast: float, probability: float) -> float:
    _check(forecast)
    return (forecast - probability) ** 2


def uncertainty(probability: float) -> float:
    return probability * (1 - probability)


def verdict(forecast: float, probability: float) -> str:
    """A one-line reaction to how close the forecast was to the truth."""
    gap = forecast - probability
    size = abs(gap)
    if size < 0.02:
        return "Spot on"
    word = "Very close" if size < 0.05 else "Close" if size < 0.10 else "Off" if size < 0.20 else "Far off"
    return f"{word}: too {'high' if gap > 0 else 'low'}"


def summarize(answers: list[Answer]) -> dict:
    """Overall scores, plus a breakdown by kind of question."""
    n = len(answers)
    if n == 0:
        return {"n": 0}
    realized = [brier(a.forecast, a.outcome) for a in answers]
    expected = [expected_brier(a.forecast, a.probability) for a in answers]
    unavoidable = [uncertainty(a.probability) for a in answers]
    excess = [excess_brier(a.forecast, a.probability) for a in answers]
    signed = [a.forecast - a.probability for a in answers]
    mean_brier = sum(realized) / n
    sd = sqrt(sum((x - mean_brier) ** 2 for x in realized) / (n - 1)) if n > 1 else 0.0

    by_kind = defaultdict(list)
    for a, e, s, r in zip(answers, excess, signed, realized, strict=True):
        by_kind[a.kind].append((e, s, r))
    kinds = [
        {"kind": kind, "n": len(rows), "mean_excess": sum(r[0] for r in rows) / len(rows),
         "mean_signed_error": sum(r[1] for r in rows) / len(rows), "mean_brier": sum(r[2] for r in rows) / len(rows)}
        for kind, rows in sorted(by_kind.items())
    ]
    mean_expected = sum(expected) / n
    return {
        "n": n,
        "mean_brier": mean_brier,
        "brier_standard_error": sd / sqrt(n) if n > 1 else None,
        "mean_expected_brier": mean_expected,
        "mean_uncertainty": sum(unavoidable) / n,  # the best score anyone could expect on these questions
        "mean_excess": sum(excess) / n,  # your reliability error: zero for a perfect forecaster
        "baseline_brier": BASELINE_BRIER,
        "skill_vs_always_50_realized": 1 - mean_brier / BASELINE_BRIER,
        "skill_vs_always_50_expected": 1 - mean_expected / BASELINE_BRIER,
        "mean_signed_error": sum(signed) / n,  # positive: you tend to guess probabilities too high
        "events_that_happened": sum(1 for a in answers if a.outcome),
        "by_kind": kinds,
    }


def calibration_bins(answers: list[Answer], bins: int = 10) -> list[dict]:
    """Group forecasts into equal-width bins. For each non-empty bin: the average forecast, the average
    TRUE probability (exact), and how often the event actually happened (noisy)."""
    grouped: dict[int, list[Answer]] = defaultdict(list)
    for a in answers:
        grouped[min(bins - 1, int(a.forecast * bins))].append(a)
    out = []
    for index in sorted(grouped):
        rows = grouped[index]
        out.append({
            "lower": index / bins, "upper": (index + 1) / bins, "n": len(rows),
            "mean_forecast": sum(a.forecast for a in rows) / len(rows),
            "mean_probability": sum(a.probability for a in rows) / len(rows),
            "observed_frequency": sum(1 for a in rows if a.outcome) / len(rows),
        })
    return out


def series(answers: list[Answer], window: int = 10) -> dict[str, list[float]]:
    """Scores as the answers come in: per answer, cumulative averages, and a rolling average."""
    realized = [brier(a.forecast, a.outcome) for a in answers]
    expected = [expected_brier(a.forecast, a.probability) for a in answers]
    unavoidable = [uncertainty(a.probability) for a in answers]

    def running(values: list[float]) -> list[float]:
        total, out = 0.0, []
        for i, v in enumerate(values, start=1):
            total += v
            out.append(total / i)
        return out

    return {
        "index": list(range(1, len(answers) + 1)),
        "brier": realized,
        "cumulative_brier": running(realized),
        "cumulative_expected": running(expected),
        "cumulative_uncertainty": running(unavoidable),
        "rolling_brier": [sum(realized[max(0, i - window + 1): i + 1]) / len(realized[max(0, i - window + 1): i + 1]) for i in range(len(realized))],
    }
