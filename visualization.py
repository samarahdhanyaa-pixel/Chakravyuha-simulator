"""Presentation helpers: battlefield HTML, risk radar, and experiment charts."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from state import ARROWS, DIRECTIONS, Observation


def battlefield_html(snapshot: dict[str, Any], *, instructor_truth: bool = False, size: int = 13) -> str:
    observation: Observation = snapshot["observation"]
    truth = snapshot["truth"] if instructor_truth else {}
    walls = truth.get("walls", observation.visible_walls)
    hazards = truth.get("hazards", observation.visible_hazards)
    enemies = truth.get("enemies", observation.visible_enemies)
    cells: list[str] = []
    next_cell = None
    action = snapshot.get("action")
    if action in DIRECTIONS:
        dr, dc = DIRECTIONS[action]
        next_cell = (observation.position[0] + dr, observation.position[1] + dc)

    for row in range(observation.rows):
        for col in range(observation.cols):
            cell = (row, col)
            visible = cell in observation.visible_cells
            classes = ["tile"]
            icon, label = "·", "Known open terrain"
            if not visible:
                classes.append("fog")
                icon, label = "?", "Unobserved terrain"
            else:
                classes.append("known")
                if cell in observation.visited_cells:
                    classes.append("visited")
            if cell in walls:
                classes.append("wall")
                icon, label = "▧", "Wall"
                if not visible and instructor_truth:
                    classes.append("truth-hidden")
            elif cell in enemies:
                classes.append("enemy")
                icon, label = "✕", "Enemy"
                if not visible and instructor_truth:
                    classes.append("truth-hidden")
            elif cell in hazards:
                classes.append("hazard")
                icon, label = "⌁", "Hazard"
                if not visible and instructor_truth:
                    classes.append("truth-hidden")
            if cell == observation.goal:
                classes.append("goal")
                icon, label = "✹", "Objective"
            if cell == observation.position:
                classes.append("agent")
                icon, label = "♙", "Agent"
            if cell == next_cell and cell != observation.position:
                classes.append("next")
            title = f"Row {row + 1}, column {col + 1} · {label}"
            cells.append(
                f'<div class="tile {" ".join(classes)}" title="{title}" aria-label="{title}">{icon}</div>'
            )

    return (
        '<div class="board-stage">'
        f'<div class="battle-grid" style="grid-template-columns:repeat({observation.cols},minmax(0,1fr))">'
        + "".join(cells)
        + "</div></div>"
    )


def uncertainty_radar(observation: Observation) -> go.Figure:
    """Chart counts of observed/unknown cells by direction, not threat probabilities."""
    values = []
    labels = []
    row, col = observation.position
    vectors = {"North": (-1, 0), "East": (0, 1), "South": (1, 0), "West": (0, -1)}
    for label, (dr, dc) in vectors.items():
        unknown = 0
        r, c = row + dr, col + dc
        while 0 <= r < observation.rows and 0 <= c < observation.cols:
            if (r, c) not in observation.visible_cells:
                unknown += 1
            r += dr
            c += dc
        labels.append(label)
        values.append(unknown)
    figure = go.Figure(
        go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            line={"color": "#c6a15b", "width": 2},
            fillcolor="rgba(198,161,91,0.16)",
            hovertemplate="%{theta}: %{r} unobserved cells<extra></extra>",
        )
    )
    figure.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "gridcolor": "rgba(198,161,91,.18)", "color": "#a99a7b", "rangemode": "tozero"},
            "angularaxis": {"gridcolor": "rgba(198,161,91,.15)", "color": "#d7c69f"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d9cfb8", "family": "Georgia, serif"},
        margin={"l": 32, "r": 32, "t": 14, "b": 20},
        height=250,
        showlegend=False,
    )
    return figure


def comparison_chart(frame: Any, metric: str, title: str, suffix: str = "") -> go.Figure:
    colors = ["#c6a15b", "#9c3434", "#7f9d87"]
    figure = go.Figure(
        go.Bar(
            x=frame["strategy"],
            y=frame[metric],
            marker_color=[colors[index % len(colors)] for index in range(len(frame))],
            text=[f"{value:.1f}{suffix}" for value in frame[metric]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.2f}" + suffix + "<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.02, "font": {"size": 16, "color": "#eadcb9"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cabd9f"},
        margin={"l": 12, "r": 10, "t": 42, "b": 34},
        height=280,
        xaxis={"gridcolor": "rgba(198,161,91,.12)"},
        yaxis={"gridcolor": "rgba(198,161,91,.12)", "zerolinecolor": "rgba(198,161,91,.2)"},
    )
    return figure


def legend_html() -> str:
    items = [
        ("agent", "♙ Agent"),
        ("goal", "✹ Objective"),
        ("enemy", "✕ Enemy"),
        ("hazard", "⌁ Hazard"),
        ("wall", "▧ Wall"),
        ("fog", "? Unknown"),
        ("visited", "· Explored"),
    ]
    return '<div class="legend">' + "".join(
        f'<span><i class="legend-{kind}">{symbol[0]}</i>{symbol[2:]}</span>' for kind, symbol in items
    ) + "</div>"
