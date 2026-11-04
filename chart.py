"""Dependency-free SVG chart: Monte Carlo error vs sample size on log-log axes.

Draws, for one ConvergenceResult:
  * the mean absolute error (blue line)
  * the 5th-95th percentile band of the error across replicates (light blue)
  * the theoretical 1/sqrt(n) reference line (red, dashed)
SVG renders in browsers and on GitHub, and can be embedded in the web app later.
"""

from math import ceil, floor, log10
from pathlib import Path
from xml.sax.saxutils import escape

from .convergence import ConvergenceResult

WIDTH, HEIGHT = 760, 500
LEFT, RIGHT, TOP, BOTTOM = 80, 30, 60, 70
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM

BLUE, BAND, RED, GRID, TEXT = "#2563eb", "#93c5fd", "#dc2626", "#e5e7eb", "#111827"


def _number_label(value: float) -> str:
    """Tick label: 0.001, 0.01, 0.1, 1, 10, 100, 1,000 ..."""
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def loglog_chart_svg(result: ConvergenceResult, title: str | None = None) -> str:
    """Return the chart as an SVG string."""
    ns = [s.n for s in result.stats]
    means = [s.mean_abs_error for s in result.stats]
    highs = [s.error_p95 for s in result.stats]
    refs = [result.reference_error(n) for n in ns]

    positive = [v for v in means + refs if v > 0]
    x_lo, x_hi = floor(log10(min(ns))), ceil(log10(max(ns)))
    y_lo = floor(log10(min(positive) / 1.5))
    y_hi = ceil(log10(max(highs + refs)))
    if x_hi == x_lo:
        x_hi += 1
    if y_hi == y_lo:
        y_hi += 1

    def px(n: float) -> float:
        return LEFT + (log10(n) - x_lo) / (x_hi - x_lo) * PLOT_W

    def py(v: float) -> float:
        v = max(v, 10.0**y_lo)  # values at or below the axis floor are drawn on the floor
        return TOP + (y_hi - log10(v)) / (y_hi - y_lo) * PLOT_H

    out: list[str] = []
    add = out.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="system-ui, sans-serif" font-size="13">'
    )
    add(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>')
    heading = title or f"Monte Carlo error vs sample size: {result.label}"
    add(f'<text x="{WIDTH / 2}" y="26" text-anchor="middle" font-size="17" '
        f'font-weight="bold" fill="{TEXT}">{escape(heading)}</text>')
    add(f'<text x="{WIDTH / 2}" y="46" text-anchor="middle" fill="#6b7280">'
        f'exact p = {result.exact:.5f}, {result.replicates} replicates, seed {result.seed}</text>')

    # grid + tick labels (one line per power of ten)
    for k in range(x_lo, x_hi + 1):
        x = px(10.0**k)
        add(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP + PLOT_H}" stroke="{GRID}"/>')
        add(f'<text x="{x:.1f}" y="{TOP + PLOT_H + 20}" text-anchor="middle" fill="{TEXT}">'
            f'{_number_label(10.0**k)}</text>')
    for k in range(y_lo, y_hi + 1):
        y = py(10.0**k)
        add(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_W}" y2="{y:.1f}" stroke="{GRID}"/>')
        add(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{TEXT}">'
            f'{_number_label(10.0**k)}</text>')
    add(f'<rect x="{LEFT}" y="{TOP}" width="{PLOT_W}" height="{PLOT_H}" fill="none" stroke="#9ca3af"/>')

    # 5th-95th percentile band
    lows = [s.error_p05 for s in result.stats]
    upper = " ".join(f"{px(n):.1f},{py(h):.1f}" for n, h in zip(ns, highs, strict=True))
    lower = " ".join(f"{px(n):.1f},{py(lo):.1f}" for n, lo in reversed(list(zip(ns, lows, strict=True))))
    add(f'<polygon points="{upper} {lower}" fill="{BAND}" fill-opacity="0.35" stroke="none"/>')

    # reference line and measured mean error
    ref_pts = " ".join(f"{px(n):.1f},{py(r):.1f}" for n, r in zip(ns, refs, strict=True))
    add(f'<polyline points="{ref_pts}" fill="none" stroke="{RED}" stroke-width="2" stroke-dasharray="7 5"/>')
    mean_pts = " ".join(f"{px(n):.1f},{py(m):.1f}" for n, m in zip(ns, means, strict=True))
    add(f'<polyline points="{mean_pts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for n, m in zip(ns, means, strict=True):
        add(f'<circle cx="{px(n):.1f}" cy="{py(m):.1f}" r="3.5" fill="{BLUE}"/>')

    # axis titles
    add(f'<text x="{LEFT + PLOT_W / 2}" y="{HEIGHT - 18}" text-anchor="middle" fill="{TEXT}">'
        f'number of simulated trials n (log scale)</text>')
    add(f'<text transform="translate(20 {TOP + PLOT_H / 2}) rotate(-90)" text-anchor="middle" '
        f'fill="{TEXT}">absolute error |estimate - exact| (log scale)</text>')

    # legend
    lx, ly = LEFT + 14, TOP + 14
    add(f'<rect x="{lx - 8}" y="{ly - 12}" width="300" height="80" fill="white" fill-opacity="0.9" stroke="#d1d5db"/>')
    add(f'<line x1="{lx}" y1="{ly}" x2="{lx + 28}" y2="{ly}" stroke="{BLUE}" stroke-width="2.5"/>')
    add(f'<text x="{lx + 36}" y="{ly + 4}" fill="{TEXT}">mean error (slope {result.slope:.2f})</text>')
    add(f'<line x1="{lx}" y1="{ly + 20}" x2="{lx + 28}" y2="{ly + 20}" stroke="{RED}" '
        f'stroke-width="2" stroke-dasharray="7 5"/>')
    add(f'<text x="{lx + 36}" y="{ly + 24}" fill="{TEXT}">theory 1/sqrt(n) (slope -0.50)</text>')
    add(f'<rect x="{lx}" y="{ly + 34}" width="28" height="10" fill="{BAND}" fill-opacity="0.6"/>')
    add(f'<text x="{lx + 36}" y="{ly + 44}" fill="{TEXT}">5th-95th percentile of error</text>')
    add(f'<text x="{lx}" y="{ly + 62}" fill="{TEXT}">95% interval coverage: {result.overall_coverage:.1%}</text>')

    add("</svg>")
    return "\n".join(out)


def write_chart(result: ConvergenceResult, path: str | Path, title: str | None = None) -> Path:
    """Write the SVG to `path` (creating folders) and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(loglog_chart_svg(result, title), encoding="utf-8")
    return target
