"""The SVG chart is valid XML, contains the right elements, and stays inside its frame."""

import xml.etree.ElementTree as ET

from app.probability import exact
from app.probability.chart import LEFT, PLOT_H, PLOT_W, TOP, loglog_chart_svg, write_chart
from app.probability.convergence import convergence_study
from app.probability.montecarlo import dice_sum_trial, log_spaced_sizes

SVG = "{http://www.w3.org/2000/svg}"
SIZES = log_spaced_sizes(2_000)
RESULT = convergence_study(
    dice_sum_trial(2, 7), float(exact.dice_sum_probability(2, 7)), SIZES, 40, seed=4, label="two dice sum to 7"
)


def parse(svg_text: str) -> ET.Element:
    return ET.fromstring(svg_text)


def test_svg_is_valid_xml_with_the_expected_parts():
    root = parse(loglog_chart_svg(RESULT))
    assert root.tag == SVG + "svg"
    assert len(root.findall(SVG + "polyline")) == 2  # measured error + 1/sqrt(n) reference
    assert len(root.findall(SVG + "polygon")) == 1  # percentile band
    assert len(root.findall(SVG + "circle")) == len(SIZES)  # one marker per sample size
    text = " ".join("".join(t.itertext()) for t in root.findall(SVG + "text"))
    assert "two dice sum to 7" in text
    assert "1/sqrt(n)" in text
    assert "coverage" in text


def test_all_drawn_points_lie_inside_the_plot_area():
    root = parse(loglog_chart_svg(RESULT))
    for shape in root.findall(SVG + "polyline") + root.findall(SVG + "polygon"):
        for pair in shape.attrib["points"].split():
            x, y = (float(v) for v in pair.split(","))
            assert LEFT - 0.5 <= x <= LEFT + PLOT_W + 0.5
            assert TOP - 0.5 <= y <= TOP + PLOT_H + 0.5


def test_measured_line_goes_down_and_to_the_right():
    root = parse(loglog_chart_svg(RESULT))
    measured = root.findall(SVG + "polyline")[1].attrib["points"].split()
    xs = [float(p.split(",")[0]) for p in measured]
    ys = [float(p.split(",")[1]) for p in measured]
    assert xs == sorted(xs)
    assert ys[-1] > ys[0]  # larger y pixel = lower on screen = smaller error


def test_special_characters_in_the_title_cannot_break_the_svg():
    svg = loglog_chart_svg(RESULT, title='<script>alert("x")</script> & more')
    root = parse(svg)  # would raise if the XML were broken
    assert not root.findall(".//" + SVG + "script")


def test_write_chart_creates_the_file_and_folders(tmp_path):
    target = tmp_path / "nested" / "folder" / "chart.svg"
    path = write_chart(RESULT, target)
    assert path.exists()
    parse(path.read_text(encoding="utf-8"))
