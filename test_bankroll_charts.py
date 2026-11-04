"""The bankroll charts: valid SVG, drawn inside the frame, and honest about edge cases."""

import re
import xml.etree.ElementTree as ET

import pytest

from app.bankroll.charts import fan_svg, histogram_svg, kelly_curve_svg
from app.bankroll.model import BetSpec
from app.bankroll.simulate import Strategy, simulate_bankroll

NS = "{http://www.w3.org/2000/svg}"
SPEC = BetSpec(0.55, 0.45, 1)


def parse(svg):
    return ET.fromstring(svg)


def numbers(points):
    return [tuple(float(v) for v in pair.split(",")) for pair in points.split()]


def inside_frame(root):
    """Every polyline/polygon vertex and every bar lies within the SVG canvas."""
    width, height = float(root.get("width")), float(root.get("height"))
    for shape in root.findall(NS + "polyline") + root.findall(NS + "polygon"):
        for x, y in numbers(shape.get("points")):
            assert -0.5 <= x <= width + 0.5 and -0.5 <= y <= height + 0.5
    for rect in root.findall(NS + "rect"):
        x, y = float(rect.get("x", 0)), float(rect.get("y", 0))  # the white background has no x/y
        assert x >= -0.5 and x + float(rect.get("width")) <= width + 1
        assert y >= -0.5 and y + float(rect.get("height")) <= height + 1


def texts(root):
    return " ".join("".join(t.itertext()) for t in root.findall(NS + "text"))


RESULTS = {
    "kelly": simulate_bankroll(SPEC, Strategy("kelly", 1), 100, 400, seed=1),
    "over": simulate_bankroll(SPEC, Strategy("kelly", 3), 100, 400, seed=1),
    "flat_ruin": simulate_bankroll(BetSpec(0.45, 0.55), Strategy("flat", 5), 100, 300, start=20, ruin_fraction=0, seed=1),
    "goal": simulate_bankroll(SPEC, Strategy("kelly", 1), 200, 300, start=1000, target=1500, seed=1),
    "no_edge": simulate_bankroll(BetSpec(0.5, 0.5), Strategy("kelly", 1), 50, 50, seed=1),
}


@pytest.mark.parametrize("name", RESULTS)
def test_the_histogram_and_fan_are_valid_svg_inside_their_frame(name):
    for make in (histogram_svg, fan_svg):
        root = parse(make(RESULTS[name]))
        assert root.tag == NS + "svg"
        inside_frame(root)


def test_histogram_content():
    root = parse(histogram_svg(RESULTS["kelly"]))
    text = texts(root)
    assert "Where the bankroll ended up" in text and "median" in text and "mean" in text and "start 1,000" in text
    bars = [r for r in root.findall(NS + "rect") if r.get("fill-opacity") == "0.8"]
    assert 1 <= len(bars) <= 40
    assert "ended with nothing" not in text


def test_the_histogram_reports_paths_that_were_wiped_out():
    result = RESULTS["flat_ruin"]
    assert result.fraction_wiped_out > 0
    text = texts(parse(histogram_svg(result)))
    assert "ended with nothing" in text and f"{round(result.fraction_wiped_out * 300):,}" in text


def test_a_histogram_of_identical_outcomes_does_not_crash():
    flat = RESULTS["no_edge"]
    assert set(flat.finals) == {1000.0}
    inside_frame(parse(histogram_svg(flat)))
    inside_frame(parse(fan_svg(flat)))


def test_the_fan_chart_shows_the_median_the_bands_and_the_ruin_level():
    root = parse(fan_svg(RESULTS["kelly"]))
    assert len(root.findall(NS + "polygon")) == 2  # 5-95 and 25-75 percentile bands
    assert len(root.findall(NS + "polyline")) >= 2 + 8  # median, start, ruin, 8 sample paths (some lines are legend swatches)
    assert "ruin level 100" in texts(root) and "25th-75th percentile" in texts(root)
    assert "ruin level" not in texts(parse(fan_svg(RESULTS["flat_ruin"])))  # ruin level 0 is the axis floor


def test_the_fan_chart_handles_paths_that_hit_zero_on_a_log_axis():
    root = parse(fan_svg(RESULTS["flat_ruin"]))
    inside_frame(root)  # zeros are pinned to the floor instead of producing -infinity


# ---------------------------------------------------------------- Kelly curve


def test_kelly_curve_marks_the_kelly_and_break_even_fractions():
    root = parse(kelly_curve_svg(SPEC))
    inside_frame(root)
    text = texts(root)
    assert "Kelly 10.0%" in text and "half Kelly 5.0%" in text and "break-even 19.9%" in text
    assert "win 55.0%, lose 45.0%" in text and "edge +0.100" in text
    curve = [p for p in root.findall(NS + "polyline") if p.get("stroke") == "#2563eb" and len(p.get("points").split()) > 100]
    assert len(curve) == 1


def test_the_peak_of_the_drawn_curve_is_at_the_kelly_fraction():
    root = parse(kelly_curve_svg(SPEC))
    curve = [p for p in root.findall(NS + "polyline") if len(p.get("points").split()) > 100][0]
    points = numbers(curve.get("points"))
    top = min(points, key=lambda xy: xy[1])  # smallest pixel y = highest growth
    left, right = points[0][0], points[-1][0]
    fraction_of_axis = (top[0] - left) / (right - left)
    x_max = SPEC.breakeven_fraction() * 1.25
    assert fraction_of_axis * x_max == pytest.approx(SPEC.kelly_fraction(), abs=0.003)


def test_kelly_curve_with_no_edge_says_so_and_does_not_crash():
    root = parse(kelly_curve_svg(BetSpec(0.5, 0.5)))
    inside_frame(root)
    text = texts(root)
    assert "No edge: the best stake is zero" in text
    assert "break-even" not in text and "half Kelly" not in text  # nothing to mark: the best stake is zero


@pytest.mark.parametrize("spec", [BetSpec(0.7, 0.0), BetSpec(0.2, 0.8, 5), BetSpec(4 / 6, 1 / 6), BetSpec(0.4, 0.6), BetSpec(0.999, 0.001, 1000)])
def test_kelly_curve_is_valid_for_unusual_bets(spec):
    inside_frame(parse(kelly_curve_svg(spec)))


def test_charts_contain_no_markup_from_numbers_or_labels():
    for make in (histogram_svg, fan_svg):
        svg = make(RESULTS["over"])
        assert "<script" not in svg and not re.search(r"on\w+=", svg)
