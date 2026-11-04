"""A tiny dependency-free SVG toolkit for the Phase 7 charts (axes, ticks, shapes, legend).

Every piece of text that reaches the SVG is XML-escaped, so a label can never break the
document or inject markup.
"""

from math import ceil, floor, log10
from xml.sax.saxutils import escape

BLUE, LIGHT_BLUE, RED, GREEN, GREY, GRID, TEXT = "#2563eb", "#93c5fd", "#dc2626", "#16a34a", "#6b7280", "#e5e7eb", "#111827"


class Axis:
    """Maps data values to a 0..1 fraction of the axis length (linear or logarithmic)."""

    def __init__(self, low: float, high: float, log: bool = False) -> None:
        if not high > low:
            raise ValueError("axis needs high > low")
        if log and low <= 0:
            raise ValueError("a log axis needs a positive lower bound")
        self.low, self.high, self.log = low, high, log

    def fraction(self, value: float) -> float:
        """Position of `value` along the axis, clamped to the axis ends."""
        if self.log:
            value = max(value, self.low)
            f = (log10(value) - log10(self.low)) / (log10(self.high) - log10(self.low))
        else:
            f = (value - self.low) / (self.high - self.low)
        return min(1.0, max(0.0, f))

    def ticks(self) -> list[float]:
        if self.log:
            first, last = floor(log10(self.low)), ceil(log10(self.high))
            multipliers = (1, 2, 5) if last - first <= 3 else (1,)
            values = [m * 10.0**k for k in range(first, last + 1) for m in multipliers]
            return [v for v in values if self.low * 0.999 <= v <= self.high * 1.001]
        span = self.high - self.low
        raw = span / 6
        magnitude = 10 ** floor(log10(raw))
        step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
        start = ceil(self.low / step - 1e-9) * step
        out, value = [], start
        while value <= self.high + step * 1e-9:
            out.append(round(value, 10))
            value += step
        return out


def format_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:g}"


class Chart:
    """One SVG chart: a titled frame with an x axis, a y axis and shapes drawn in data coordinates."""

    def __init__(self, x: Axis, y: Axis, title: str, subtitle: str = "", x_label: str = "", y_label: str = "",
                 width: int = 760, height: int = 480, left: int = 80, right: int = 30, top: int = 60, bottom: int = 70):
        self.xa, self.ya = x, y
        self.width, self.height = width, height
        self.left, self.top = left, top
        self.plot_w, self.plot_h = width - left - right, height - top - bottom
        self.parts: list[str] = []
        self.title, self.subtitle, self.x_label, self.y_label = title, subtitle, x_label, y_label

    # ---- coordinates
    def px(self, value: float) -> float:
        return self.left + self.xa.fraction(value) * self.plot_w

    def py(self, value: float) -> float:
        return self.top + (1 - self.ya.fraction(value)) * self.plot_h

    def point(self, x: float, y: float) -> str:
        return f"{self.px(x):.1f},{self.py(y):.1f}"

    # ---- shapes (data coordinates)
    def grid(self) -> None:
        for v in self.xa.ticks():
            x = self.px(v)
            self.parts.append(f'<line x1="{x:.1f}" y1="{self.top}" x2="{x:.1f}" y2="{self.top + self.plot_h}" stroke="{GRID}"/>')
            self.parts.append(f'<text x="{x:.1f}" y="{self.top + self.plot_h + 20}" text-anchor="middle" fill="{TEXT}">{escape(format_number(v))}</text>')
        for v in self.ya.ticks():
            y = self.py(v)
            self.parts.append(f'<line x1="{self.left}" y1="{y:.1f}" x2="{self.left + self.plot_w}" y2="{y:.1f}" stroke="{GRID}"/>')
            self.parts.append(f'<text x="{self.left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{TEXT}">{escape(format_number(v))}</text>')
        self.parts.append(f'<rect x="{self.left}" y="{self.top}" width="{self.plot_w}" height="{self.plot_h}" fill="none" stroke="#9ca3af"/>')

    def polyline(self, points: list[tuple[float, float]], color: str, width: float = 2.0, dash: str | None = None, opacity: float = 1.0) -> None:
        if len(points) < 2:
            return
        pts = " ".join(self.point(x, y) for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}"{extra}/>')

    def band(self, xs: list[float], lower: list[float], upper: list[float], color: str, opacity: float) -> None:
        top = " ".join(self.point(x, v) for x, v in zip(xs, upper, strict=True))
        bottom = " ".join(self.point(x, v) for x, v in reversed(list(zip(xs, lower, strict=True))))
        self.parts.append(f'<polygon points="{top} {bottom}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>')

    def vline(self, x: float, color: str, dash: str | None = "6 4", width: float = 1.5) -> None:
        self.polyline([(x, self.ya.low), (x, self.ya.high)], color, width, dash)

    def hline(self, y: float, color: str, dash: str | None = "6 4", width: float = 1.5) -> None:
        self.polyline([(self.xa.low, y), (self.xa.high, y)], color, width, dash)

    def bar(self, x0: float, x1: float, y0: float, y1: float, color: str, opacity: float = 0.8) -> None:
        left, right = self.px(x0), self.px(x1)
        top, bottom = self.py(y1), self.py(y0)
        self.parts.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{max(right - left - 1, 0.5):.1f}" height="{max(bottom - top, 0):.1f}" fill="{color}" fill-opacity="{opacity}"/>')

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str | None = None, opacity: float = 0.9) -> None:
        stroke_attr = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
        self.parts.append(f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="{opacity}"{stroke_attr}/>')

    def label(self, x: float, y: float, text: str, anchor: str = "start", color: str = TEXT) -> None:
        self.parts.append(f'<text x="{self.px(x):.1f}" y="{self.py(y):.1f}" text-anchor="{anchor}" fill="{color}">{escape(text)}</text>')

    def legend(self, items: list[tuple[str, str, str | None]], x: int | None = None, y: int | None = None) -> None:
        """items: (label, color, dash). Drawn in the top-left of the plot unless x/y (pixels) are given."""
        x = self.left + 14 if x is None else x
        y = self.top + 14 if y is None else y
        height = 12 + 20 * len(items)
        width = 16 + 8 * max(len(label) for label, _, _ in items) + 40
        self.parts.append(f'<rect x="{x - 8}" y="{y - 12}" width="{width}" height="{height}" fill="white" fill-opacity="0.9" stroke="#d1d5db"/>')
        for i, (label, color, dash) in enumerate(items):
            yy = y + 20 * i
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            self.parts.append(f'<line x1="{x}" y1="{yy}" x2="{x + 28}" y2="{yy}" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
            self.parts.append(f'<text x="{x + 36}" y="{yy + 4}" fill="{TEXT}">{escape(label)}</text>')

    def to_svg(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="{self.width}" '
            f'height="{self.height}" font-family="system-ui, sans-serif" font-size="13">'
            f'<rect width="{self.width}" height="{self.height}" fill="white"/>'
            f'<text x="{self.width / 2}" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="{TEXT}">{escape(self.title)}</text>'
        )
        if self.subtitle:
            head += f'<text x="{self.width / 2}" y="46" text-anchor="middle" fill="{GREY}">{escape(self.subtitle)}</text>'
        foot = ""
        if self.x_label:
            foot += f'<text x="{self.left + self.plot_w / 2}" y="{self.height - 18}" text-anchor="middle" fill="{TEXT}">{escape(self.x_label)}</text>'
        if self.y_label:
            foot += f'<text transform="translate(20 {self.top + self.plot_h / 2}) rotate(-90)" text-anchor="middle" fill="{TEXT}">{escape(self.y_label)}</text>'
        return head + "".join(self.parts) + foot + "</svg>"
