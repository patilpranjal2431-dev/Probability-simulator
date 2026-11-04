"""The two calibration charts: valid SVG, inside their frame, sensible for tiny and large inputs."""

import random
import xml.etree.ElementTree as ET

import pytest

from app.calibration.charts import brier_over_time_svg, reliability_svg
from app.calibration.scoring import Answer, calibration_bins, series

NS = "{http://www.w3.org/2000/svg}"


def make_answers(n, seed=1, noise=0.0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = rng.uniform(0.05, 0.95)
        out.append(Answer("duel_win", min(1, max(0, p + rng.uniform(-noise, noise))), p, rng.random() < p))
    return out


def inside_frame(root):
    width, height = float(root.get("width")), float(root.get("height"))
    for shape in root.findall(NS + "polyline") + root.findall(NS + "polygon"):
        for pair in shape.get("points").split():
            x, y = (float(v) for v in pair.split(","))
            assert -0.5 <= x <= width + 0.5 and -0.5 <= y <= height + 0.5
    for circle in root.findall(NS + "circle"):
        assert 0 <= float(circle.get("cx")) <= width and 0 <= float(circle.get("cy")) <= height


def texts(root):
    return " ".join("".join(t.itertext()) for t in root.findall(NS + "text"))


@pytest.mark.parametrize("n", [1, 2, 7, 60, 400])
def test_the_brier_chart_is_valid_for_any_number_of_answers(n):
    root = ET.fromstring(brier_over_time_svg(series(make_answers(n))))
    inside_frame(root)
    text = texts(root)
    assert "Your Brier score over time" in text and "always saying 50%" in text and "a perfect forecaster would score" in text


def test_the_brier_chart_draws_the_reference_and_data_lines():
    root = ET.fromstring(brier_over_time_svg(series(make_answers(30))))
    colours = [p.get("stroke") for p in root.findall(NS + "polyline")]
    assert "#6b7280" in colours and "#16a34a" in colours and "#2563eb" in colours  # 50% line, perfect line, your line


def test_the_brier_chart_shows_a_dot_per_answer_only_when_there_are_few():
    few = ET.fromstring(brier_over_time_svg(series(make_answers(10))))
    many = ET.fromstring(brier_over_time_svg(series(make_answers(200))))
    assert len(few.findall(NS + "circle")) == 10 and len(many.findall(NS + "circle")) == 0


@pytest.mark.parametrize("n", [1, 5, 80, 500])
def test_the_reliability_chart_is_valid_for_any_number_of_answers(n):
    root = ET.fromstring(reliability_svg(calibration_bins(make_answers(n, noise=0.2))))
    inside_frame(root)
    assert "Calibration" in texts(root) and "perfect calibration" in texts(root)


def test_the_reliability_chart_has_two_dots_per_bin_and_the_diagonal():
    bins = calibration_bins(make_answers(300, noise=0.1))
    root = ET.fromstring(reliability_svg(bins))
    assert len(root.findall(NS + "circle")) == 2 * len(bins)  # exact truth + what actually happened
    diagonals = [p for p in root.findall(NS + "polyline") if p.get("stroke-dasharray") and p.get("stroke") == "#6b7280"]
    assert len(diagonals) >= 1


def test_extreme_forecasts_are_drawn_inside_the_frame():
    answers = [Answer("poker", 0.0, 0.4, True), Answer("poker", 1.0, 0.4, False), Answer("poker", 1.0, 0.4, True)]
    inside_frame(ET.fromstring(reliability_svg(calibration_bins(answers))))
    inside_frame(ET.fromstring(brier_over_time_svg(series(answers))))
