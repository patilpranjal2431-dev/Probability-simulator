"""SVG charts for the calibration game."""

from math import sqrt

from app.svgkit import BLUE, GREEN, GREY, LIGHT_BLUE, RED, Axis, Chart

from .scoring import BASELINE_BRIER


def brier_over_time_svg(data: dict) -> str:
    """Your Brier score as the answers come in, against two reference lines."""
    n = len(data["index"])
    x_axis = Axis(0.5, 1.5) if n == 1 else Axis(1, n)
    lines = data["cumulative_brier"] + data["cumulative_expected"] + data["cumulative_uncertainty"]
    top = max(0.3, max(lines) * 1.1)
    chart = Chart(x_axis, Axis(0, top), "Your Brier score over time",
                  "lower is better: 0.25 is what always saying 50% scores", "questions answered", "average Brier score")
    chart.grid()
    xs = [float(i) for i in data["index"]]
    chart.hline(BASELINE_BRIER, GREY)
    chart.polyline(list(zip(xs, data["cumulative_uncertainty"], strict=True)), GREEN, 2.0, "6 4")
    chart.polyline(list(zip(xs, data["cumulative_brier"], strict=True)), LIGHT_BLUE, 2.0)
    chart.polyline(list(zip(xs, data["cumulative_expected"], strict=True)), BLUE, 3.0)
    for x, y in zip(xs, data["cumulative_expected"], strict=True):
        if n <= 40:
            chart.circle(x, y, 3.5, BLUE)
    chart.legend([
        ("your average (noise-free expected score)", BLUE, None),
        ("your average (actual outcomes, noisy)", LIGHT_BLUE, None),
        ("a perfect forecaster would score", GREEN, "6 4"),
        ("always saying 50%", GREY, "6 4"),
    ], y=chart.top + chart.plot_h - 80)
    return chart.to_svg()


def reliability_svg(bins: list[dict]) -> str:
    """The calibration (reliability) diagram. Each dot is a group of your forecasts: where it sits is the
    average forecast (across) against the average TRUE probability (up). On the dashed diagonal means
    well calibrated; above it means you under-estimated, below it means you over-estimated."""
    chart = Chart(Axis(0, 1), Axis(0, 1), "Calibration: what you said vs what was true",
                  "on the diagonal = well calibrated; dot size = how many forecasts", "your forecast (bin average)",
                  "true probability / how often it happened")
    chart.grid()
    chart.polyline([(0, 0), (1, 1)], GREY, 1.5, "6 4")
    points = [(b["mean_forecast"], b["mean_probability"]) for b in bins]
    chart.polyline(points, BLUE, 2.0, None, 0.6)
    for b in bins:
        radius = min(12.0, 4.0 + 2.0 * sqrt(b["n"]))
        chart.circle(b["mean_forecast"], b["observed_frequency"], radius * 0.8, "white", RED, 0.9)
        chart.circle(b["mean_forecast"], b["mean_probability"], radius * 0.6, BLUE)
    chart.legend([
        ("true probability (exact)", BLUE, None),
        ("how often it actually happened", RED, None),
        ("perfect calibration", GREY, "6 4"),
    ], y=chart.top + chart.plot_h - 60)
    return chart.to_svg()
