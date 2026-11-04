"""The shared SVG toolkit."""

import xml.etree.ElementTree as ET

import pytest

from app.svgkit import Axis, Chart, format_number


def test_linear_axis_maps_and_clamps():
    axis = Axis(0, 10)
    assert axis.fraction(0) == 0 and axis.fraction(5) == 0.5 and axis.fraction(10) == 1
    assert axis.fraction(-5) == 0 and axis.fraction(99) == 1  # clamped to the ends


def test_log_axis_maps_decades_evenly_and_clamps_non_positive_values():
    axis = Axis(1, 10_000, log=True)
    assert axis.fraction(10) == pytest.approx(0.25) and axis.fraction(100) == pytest.approx(0.5)
    assert axis.fraction(0) == 0 and axis.fraction(-3) == 0  # cannot be drawn: pinned to the floor


def test_bad_axes_are_rejected():
    with pytest.raises(ValueError):
        Axis(5, 5)
    with pytest.raises(ValueError):
        Axis(0, 10, log=True)


def test_ticks_are_inside_the_range_and_sensible():
    assert Axis(10, 10_000, log=True).ticks() == [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000]
    assert Axis(0, 0.35).ticks() == [0.0, 0.1, 0.2, 0.3]
    for axis in (Axis(0, 1), Axis(-0.007, 0.006), Axis(3, 17), Axis(80, 1500, log=True)):
        ticks = axis.ticks()
        assert ticks and all(axis.low * (0.999 if axis.low > 0 else 1) - 1e-9 <= t <= axis.high * 1.001 + 1e-9 for t in ticks)


def test_number_formatting():
    assert [format_number(v) for v in (0, 0.005, 12.4, 1234.5, 2_500_000)] == ["0", "0.005", "12", "1,234", "2,500,000"]


def test_text_is_escaped_so_a_label_cannot_break_or_inject_into_the_svg():
    chart = Chart(Axis(0, 1), Axis(0, 1), '<script>alert("x")</script> & title', "<b>sub</b>", "x<y", "a&b")
    chart.grid()
    chart.label(0.5, 0.5, "<img src=x onerror=alert(1)>")
    chart.legend([("<bad>", "#000000", None)])
    svg = chart.to_svg()
    root = ET.fromstring(svg)  # would raise if the XML were broken
    assert not root.findall(".//{http://www.w3.org/2000/svg}script")
    assert "<script>" not in svg and "<img" not in svg and "<b>" not in svg


def test_shapes_land_inside_the_plot_area():
    chart = Chart(Axis(0, 10), Axis(1, 1000, log=True), "t")
    for x, y in ((0, 1), (10, 1000), (5, 0.001), (50, 1e9)):  # includes out-of-range values
        px, py = chart.px(x), chart.py(y)
        assert chart.left - 1e-9 <= px <= chart.left + chart.plot_w + 1e-9
        assert chart.top - 1e-9 <= py <= chart.top + chart.plot_h + 1e-9
