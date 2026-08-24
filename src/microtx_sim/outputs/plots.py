"""Dependency-free deterministic SVG charts for reproducible batch reports.

Render functions return complete SVG text.  Write functions atomically persist
that text.  Frontier rows require ``scenario``, ``producer_revenue_cents`` and
``mean_harm`` by default; decomposition rows require ``component`` and
``value``; EPGC rows require ``scenario`` and
``minimum_public_contribution_cents``.  Keyword arguments can select equivalent
columns without coupling the output layer to future batch dataclasses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
import math
from pathlib import Path

from .writers import write_text_atomic


_WIDTH = 800
_HEIGHT = 480
_LEFT = 82.0
_RIGHT = 28.0
_TOP = 58.0
_BOTTOM = 70.0
_PLOT_WIDTH = _WIDTH - _LEFT - _RIGHT
_PLOT_HEIGHT = _HEIGHT - _TOP - _BOTTOM


def render_harm_distribution_svg(
    values: Sequence[float], *, bins: int = 20, title: str = "Harm distribution"
) -> str:
    """Render a histogram of finite harm scores; empty input is valid."""

    return _render_histogram(
        values,
        bins=bins,
        title=title,
        x_label="Composite harm score",
        description="Distribution of simulated player harm scores.",
        fill="#4C78A8",
    )


def render_spending_distribution_svg(
    values: Sequence[float],
    *,
    bins: int = 20,
    title: str = "Spending distribution",
) -> str:
    """Render a histogram of spending in simulation cents; zeros are retained."""

    return _render_histogram(
        values,
        bins=bins,
        title=title,
        x_label="Player spending (simulation cents)",
        description="Distribution of simulated player spending.",
        fill="#F58518",
    )


def render_harm_revenue_frontier_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    scenario_key: str = "scenario",
    revenue_key: str = "producer_revenue_cents",
    harm_key: str = "mean_harm",
    title: str = "Harm versus revenue frontier",
) -> str:
    """Render scenario points and the non-dominated harm/revenue frontier.

    Revenue is treated as desirable and harm as undesirable. A point is on the
    frontier when no other scenario has at least as much revenue and no more
    harm, with one strict improvement.
    """

    points: list[tuple[float, float, str]] = []
    for index, row in enumerate(_mapping_rows(rows)):
        scenario = _required_label(row, scenario_key, index)
        revenue = _required_number(row, revenue_key, index)
        harm = _required_number(row, harm_key, index)
        points.append((revenue, harm, scenario))
    points.sort(key=lambda item: (item[0], item[1], item[2]))

    body = _axes("Producer revenue (simulation cents)", "Mean harm")
    if not points:
        body.append(_no_data())
    else:
        x_low, x_high = _padded_extent([point[0] for point in points])
        y_low, y_high = _padded_extent([point[1] for point in points])
        coordinates = [
            (
                _scale(revenue, x_low, x_high, _LEFT, _LEFT + _PLOT_WIDTH),
                _scale(harm, y_low, y_high, _TOP + _PLOT_HEIGHT, _TOP),
                scenario,
                revenue,
                harm,
            )
            for revenue, harm, scenario in points
        ]
        efficient = []
        for candidate in coordinates:
            _, _, _, candidate_revenue, candidate_harm = candidate
            dominated = any(
                other_revenue >= candidate_revenue
                and other_harm <= candidate_harm
                and (
                    other_revenue > candidate_revenue
                    or other_harm < candidate_harm
                )
                for _, _, _, other_revenue, other_harm in coordinates
            )
            if not dominated:
                efficient.append(candidate)
        efficient.sort(key=lambda item: (item[3], item[4], item[2]))
        if len(efficient) > 1:
            path = " ".join(
                f"{_coordinate(x)},{_coordinate(y)}" for x, y, *_ in efficient
            )
            body.append(
                f'<polyline points="{path}" fill="none" stroke="#2F6B2F" '
                'stroke-width="2"><title>Pareto-efficient frontier</title></polyline>'
            )
        for x, y, scenario, revenue, harm in coordinates:
            safe_label = escape(scenario, quote=True)
            on_frontier = any(item[2] == scenario for item in efficient)
            fill = "#54A24B" if on_frontier else "#BAB0AC"
            body.append(
                f'<circle cx="{_coordinate(x)}" cy="{_coordinate(y)}" r="5" '
                f'fill="{fill}" stroke="#244A20" stroke-width="1">'
                f'<title>{safe_label}: revenue {_number(revenue)}, harm {_number(harm)}</title>'
                "</circle>"
            )
            body.append(
                f'<text x="{_coordinate(x + 7)}" y="{_coordinate(y - 7)}" '
                f'class="label">{safe_label}</text>'
            )
        body.extend(_extent_labels(x_low, x_high, y_low, y_high))
    return _svg_document(
        title,
        "Scenario-level relationship between producer revenue and simulated harm.",
        body,
    )


def render_opportunity_cost_decomposition_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    component_key: str = "component",
    value_key: str = "value",
    title: str = "Opportunity-cost decomposition",
) -> str:
    """Render non-negative burden components as deterministic horizontal bars."""

    values: list[tuple[str, float]] = []
    for index, row in enumerate(_mapping_rows(rows)):
        label = _required_label(row, component_key, index)
        value = _required_number(row, value_key, index)
        if value < 0.0:
            raise ValueError("opportunity-cost components must be non-negative")
        values.append((label, value))
    values.sort(key=lambda item: (item[0], item[1]))

    body = _axes("Burden score or monetary proxy", "Component")
    if not values:
        body.append(_no_data())
    else:
        maximum = max((value for _, value in values), default=0.0)
        scale_max = maximum if maximum > 0.0 else 1.0
        slot = _PLOT_HEIGHT / len(values)
        bar_height = min(34.0, slot * 0.64)
        for index, (label, value) in enumerate(values):
            y = _TOP + index * slot + (slot - bar_height) / 2.0
            width = _PLOT_WIDTH * value / scale_max
            safe_label = escape(label, quote=True)
            body.append(
                f'<rect x="{_coordinate(_LEFT)}" y="{_coordinate(y)}" '
                f'width="{_coordinate(width)}" height="{_coordinate(bar_height)}" '
                'fill="#B279A2"><title>'
                f"{safe_label}: {_number(value)}</title></rect>"
            )
            body.append(
                f'<text x="{_coordinate(_LEFT - 8)}" '
                f'y="{_coordinate(y + bar_height / 2 + 4)}" '
                f'class="tick" text-anchor="end">{safe_label}</text>'
            )
            body.append(
                f'<text x="{_coordinate(_LEFT + width + 6)}" '
                f'y="{_coordinate(y + bar_height / 2 + 4)}" '
                f'class="label">{escape(_number(value))}</text>'
            )
        body.extend(_horizontal_extent_labels(scale_max))
    return _svg_document(
        title,
        "Decomposition of simulated opportunity-cost burden by displaced activity.",
        body,
    )


def render_epgc_subsidy_requirement_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    scenario_key: str = "scenario",
    subsidy_key: str = "minimum_public_contribution_cents",
    title: str = "EPGC subsidy requirement",
) -> str:
    """Render minimum public contributions for safe-game scenarios."""

    values: list[tuple[str, float]] = []
    for index, row in enumerate(_mapping_rows(rows)):
        scenario = _required_label(row, scenario_key, index)
        subsidy = _required_number(row, subsidy_key, index)
        if subsidy < 0.0:
            raise ValueError("EPGC subsidy requirements must be non-negative")
        values.append((scenario, subsidy))
    values.sort(key=lambda item: (item[0], item[1]))

    body = _axes("Scenario", "Minimum public contribution (simulation cents)")
    if not values:
        body.append(_no_data())
    else:
        maximum = max((value for _, value in values), default=0.0)
        scale_max = maximum if maximum > 0.0 else 1.0
        slot = _PLOT_WIDTH / len(values)
        bar_width = min(68.0, slot * 0.62)
        for index, (label, value) in enumerate(values):
            height = _PLOT_HEIGHT * value / scale_max
            x = _LEFT + index * slot + (slot - bar_width) / 2.0
            y = _TOP + _PLOT_HEIGHT - height
            safe_label = escape(label, quote=True)
            body.append(
                f'<rect x="{_coordinate(x)}" y="{_coordinate(y)}" '
                f'width="{_coordinate(bar_width)}" height="{_coordinate(height)}" '
                'fill="#E45756"><title>'
                f"{safe_label}: {_number(value)}</title></rect>"
            )
            body.append(
                f'<text x="{_coordinate(x + bar_width / 2)}" '
                f'y="{_coordinate(_TOP + _PLOT_HEIGHT + 18)}" class="tick" '
                f'text-anchor="middle">{safe_label}</text>'
            )
            body.append(
                f'<text x="{_coordinate(x + bar_width / 2)}" '
                f'y="{_coordinate(max(_TOP + 12, y - 6))}" class="label" '
                f'text-anchor="middle">{escape(_number(value))}</text>'
            )
        body.extend(_vertical_extent_labels(scale_max))
    return _svg_document(
        title,
        "Minimum simulated public contribution required for non-negative safe profit.",
        body,
    )


def write_harm_distribution_svg(
    path: str | Path,
    values: Sequence[float],
    *,
    bins: int = 20,
    title: str = "Harm distribution",
) -> Path:
    return write_text_atomic(
        path, render_harm_distribution_svg(values, bins=bins, title=title)
    )


def write_spending_distribution_svg(
    path: str | Path,
    values: Sequence[float],
    *,
    bins: int = 20,
    title: str = "Spending distribution",
) -> Path:
    return write_text_atomic(
        path, render_spending_distribution_svg(values, bins=bins, title=title)
    )


def write_harm_revenue_frontier_svg(
    path: str | Path, rows: Sequence[Mapping[str, object]], **kwargs: object
) -> Path:
    return write_text_atomic(path, render_harm_revenue_frontier_svg(rows, **kwargs))


def write_opportunity_cost_decomposition_svg(
    path: str | Path, rows: Sequence[Mapping[str, object]], **kwargs: object
) -> Path:
    return write_text_atomic(
        path, render_opportunity_cost_decomposition_svg(rows, **kwargs)
    )


def write_epgc_subsidy_requirement_svg(
    path: str | Path, rows: Sequence[Mapping[str, object]], **kwargs: object
) -> Path:
    return write_text_atomic(path, render_epgc_subsidy_requirement_svg(rows, **kwargs))


def _render_histogram(
    values: Sequence[float],
    *,
    bins: int,
    title: str,
    x_label: str,
    description: str,
    fill: str,
) -> str:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("histogram bins must be a positive integer")
    numbers = _number_sequence(values)
    body = _axes(x_label, "Players")
    if not numbers:
        body.append(_no_data())
        return _svg_document(title, description, body)

    low = min(numbers)
    high = max(numbers)
    if low == high:
        padding = max(abs(low) * 0.05, 0.5)
        low -= padding
        high += padding
    counts = [0] * bins
    for value in numbers:
        position = int((value - low) / (high - low) * bins)
        counts[min(bins - 1, max(0, position))] += 1
    maximum = max(counts) or 1
    slot = _PLOT_WIDTH / bins
    bar_width = max(0.0, slot - min(2.0, slot * 0.12))
    for index, count in enumerate(counts):
        height = _PLOT_HEIGHT * count / maximum
        x = _LEFT + index * slot + (slot - bar_width) / 2.0
        y = _TOP + _PLOT_HEIGHT - height
        body.append(
            f'<rect x="{_coordinate(x)}" y="{_coordinate(y)}" '
            f'width="{_coordinate(bar_width)}" height="{_coordinate(height)}" '
            f'fill="{fill}"><title>Bin {index + 1}: {count}</title></rect>'
        )
    body.extend(_histogram_extent_labels(low, high, maximum))
    return _svg_document(title, description, body)


def _svg_document(title: str, description: str, body: Sequence[str]) -> str:
    safe_title = escape(str(title), quote=True)
    safe_description = escape(description, quote=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" '
            f'height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
            'role="img" aria-labelledby="chart-title chart-description">'
        ),
        f'<title id="chart-title">{safe_title}</title>',
        f'<desc id="chart-description">{safe_description}</desc>',
        "<style>",
        ".title{font:600 20px sans-serif;fill:#222}",
        ".axis{font:12px sans-serif;fill:#333}",
        ".tick{font:11px sans-serif;fill:#444}",
        ".label{font:10px sans-serif;fill:#222}",
        "</style>",
        '<rect width="800" height="480" fill="#FFFFFF"/>',
        f'<text x="{_WIDTH / 2:g}" y="31" class="title" text-anchor="middle">{safe_title}</text>',
        *body,
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def _axes(x_label: str, y_label: str) -> list[str]:
    plot_bottom = _TOP + _PLOT_HEIGHT
    safe_x = escape(x_label, quote=True)
    safe_y = escape(y_label, quote=True)
    return [
        f'<line x1="{_coordinate(_LEFT)}" y1="{_coordinate(plot_bottom)}" '
        f'x2="{_coordinate(_LEFT + _PLOT_WIDTH)}" y2="{_coordinate(plot_bottom)}" '
        'stroke="#333" stroke-width="1"/>',
        f'<line x1="{_coordinate(_LEFT)}" y1="{_coordinate(_TOP)}" '
        f'x2="{_coordinate(_LEFT)}" y2="{_coordinate(plot_bottom)}" '
        'stroke="#333" stroke-width="1"/>',
        f'<text x="{_coordinate(_LEFT + _PLOT_WIDTH / 2)}" y="{_HEIGHT - 18}" '
        f'class="axis" text-anchor="middle">{safe_x}</text>',
        f'<text x="18" y="{_coordinate(_TOP + _PLOT_HEIGHT / 2)}" '
        f'class="axis" text-anchor="middle" transform="rotate(-90 18 '
        f'{_coordinate(_TOP + _PLOT_HEIGHT / 2)})">{safe_y}</text>',
    ]


def _histogram_extent_labels(low: float, high: float, maximum: int) -> list[str]:
    bottom = _TOP + _PLOT_HEIGHT
    return [
        f'<text x="{_coordinate(_LEFT)}" y="{_coordinate(bottom + 18)}" '
        f'class="tick" text-anchor="start">{escape(_number(low))}</text>',
        f'<text x="{_coordinate(_LEFT + _PLOT_WIDTH)}" y="{_coordinate(bottom + 18)}" '
        f'class="tick" text-anchor="end">{escape(_number(high))}</text>',
        f'<text x="{_coordinate(_LEFT - 8)}" y="{_coordinate(_TOP + 4)}" '
        f'class="tick" text-anchor="end">{maximum}</text>',
        f'<text x="{_coordinate(_LEFT - 8)}" y="{_coordinate(bottom + 4)}" '
        'class="tick" text-anchor="end">0</text>',
    ]


def _extent_labels(
    x_low: float, x_high: float, y_low: float, y_high: float
) -> list[str]:
    bottom = _TOP + _PLOT_HEIGHT
    return [
        f'<text x="{_coordinate(_LEFT)}" y="{_coordinate(bottom + 18)}" class="tick">{escape(_number(x_low))}</text>',
        f'<text x="{_coordinate(_LEFT + _PLOT_WIDTH)}" y="{_coordinate(bottom + 18)}" class="tick" text-anchor="end">{escape(_number(x_high))}</text>',
        f'<text x="{_coordinate(_LEFT - 8)}" y="{_coordinate(_TOP + 4)}" class="tick" text-anchor="end">{escape(_number(y_high))}</text>',
        f'<text x="{_coordinate(_LEFT - 8)}" y="{_coordinate(bottom + 4)}" class="tick" text-anchor="end">{escape(_number(y_low))}</text>',
    ]


def _horizontal_extent_labels(maximum: float) -> list[str]:
    bottom = _TOP + _PLOT_HEIGHT
    return [
        f'<text x="{_coordinate(_LEFT)}" y="{_coordinate(bottom + 18)}" class="tick">0</text>',
        f'<text x="{_coordinate(_LEFT + _PLOT_WIDTH)}" y="{_coordinate(bottom + 18)}" class="tick" text-anchor="end">{escape(_number(maximum))}</text>',
    ]


def _vertical_extent_labels(maximum: float) -> list[str]:
    bottom = _TOP + _PLOT_HEIGHT
    return [
        f'<text x="{_coordinate(_LEFT - 8)}" y="{_coordinate(_TOP + 4)}" class="tick" text-anchor="end">{escape(_number(maximum))}</text>',
        f'<text x="{_coordinate(_LEFT - 8)}" y="{_coordinate(bottom + 4)}" class="tick" text-anchor="end">0</text>',
    ]


def _no_data() -> str:
    return (
        f'<text x="{_coordinate(_LEFT + _PLOT_WIDTH / 2)}" '
        f'y="{_coordinate(_TOP + _PLOT_HEIGHT / 2)}" class="axis" '
        'text-anchor="middle">No data</text>'
    )


def _mapping_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    if isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("plot rows must be a sequence of mappings")
    result: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"plot row {index} is not a mapping")
        result.append(row)
    return result


def _number_sequence(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("plot values must be a sequence of numbers")
    result: list[float] = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"plot value {index} is not numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"plot value {index} must be finite")
        result.append(number)
    return result


def _required_label(row: Mapping[str, object], key: str, index: int) -> str:
    if key not in row:
        raise ValueError(f"plot row {index} is missing {key!r}")
    label = str(row[key])
    if not label:
        raise ValueError(f"plot row {index} has an empty {key!r}")
    return label


def _required_number(row: Mapping[str, object], key: str, index: int) -> float:
    if key not in row:
        raise ValueError(f"plot row {index} is missing {key!r}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"plot row {index} has non-numeric {key!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"plot row {index} has non-finite {key!r}")
    return value


def _padded_extent(values: Sequence[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    if low == high:
        padding = max(abs(low) * 0.05, 0.5)
    else:
        padding = (high - low) * 0.06
    return low - padding, high + padding


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    return start + (value - low) / (high - low) * (end - start)


def _coordinate(value: float) -> str:
    rounded = f"{value:.3f}".rstrip("0").rstrip(".")
    return rounded if rounded not in ("", "-0") else "0"


def _number(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.6g}"


__all__ = [
    "render_epgc_subsidy_requirement_svg",
    "render_harm_distribution_svg",
    "render_harm_revenue_frontier_svg",
    "render_opportunity_cost_decomposition_svg",
    "render_spending_distribution_svg",
    "write_epgc_subsidy_requirement_svg",
    "write_harm_distribution_svg",
    "write_harm_revenue_frontier_svg",
    "write_opportunity_cost_decomposition_svg",
    "write_spending_distribution_svg",
]
