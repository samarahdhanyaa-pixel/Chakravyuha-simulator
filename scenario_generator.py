"""Seeded battlefield generation with a protected, connected route to the objective."""

from __future__ import annotations

import random

from state import Coord, WorldState

PRESETS = {
    "Easy": {"wall_rate": 0.11, "hazard_count": 5, "enemy_count": 1, "max_steps": 90},
    "Medium": {"wall_rate": 0.18, "hazard_count": 9, "enemy_count": 2, "max_steps": 80},
    "Hard": {"wall_rate": 0.25, "hazard_count": 13, "enemy_count": 3, "max_steps": 72},
}

SCENARIOS = {
    "The Outer Gate": {
        "description": "A guarded southern entrance opens toward the central command point.",
        "seed": 17,
        "difficulty": "Easy",
        "visibility": "High",
        "start": (12, 1),
        "goal": (6, 6),
    },
    "Veil of Ash": {
        "description": "A smoke-veiled eastern approach bends toward a command post in the northwest ring.",
        "seed": 42,
        "difficulty": "Medium",
        "visibility": "Partial",
        "start": (12, 11),
        "goal": (3, 3),
    },
    "The Silent Wheel": {
        "description": "From a northern breach, severe fog hides the southern-western objective.",
        "seed": 93,
        "difficulty": "Hard",
        "visibility": "Severe",
        "start": (0, 11),
        "goal": (9, 3),
    },
}

SIZE = 13
START: Coord = (12, 1)
GOAL: Coord = (6, 6)


def _guaranteed_route(rng: random.Random) -> list[Coord]:
    """Generate a monotone, winding route from the outer ring to the center."""
    vertical = ["N"] * (START[0] - GOAL[0])
    horizontal = ["E"] * (GOAL[1] - START[1])
    steps: list[str] = []
    while vertical or horizontal:
        choose_horizontal = bool(horizontal) and (not vertical or rng.random() < 0.48)
        steps.append(horizontal.pop() if choose_horizontal else vertical.pop())

    route = [START]
    row, col = START
    for action in steps:
        if action == "N":
            row -= 1
        else:
            col += 1
        route.append((row, col))
    return route


def generate_scenario(
    difficulty: str = "Medium",
    seed: int = 42,
    rows: int = SIZE,
    cols: int = SIZE,
    start: Coord | None = None,
    goal: Coord | None = None,
) -> WorldState:
    """Create a reproducible scenario. The route is open; hidden truth stays in WorldState."""
    if difficulty not in PRESETS:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    if rows < 7 or cols < 7:
        raise ValueError("The battlefield must be at least 7 by 7.")

    rng = random.Random(int(seed))
    start = start or (rows - 1, 1)
    goal = goal or (rows // 2, cols // 2)
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError(f"Start coordinate is outside the battlefield: {start}")
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError(f"Goal coordinate is outside the battlefield: {goal}")
    if start == goal:
        raise ValueError("Start and objective must be different cells.")
    route_rng = random.Random(int(seed) ^ 0xC0FFEE)
    route = _route_for(start, goal, route_rng)
    route_cells = set(route)
    protected = {
        (r + dr, c + dc)
        for r, c in route_cells
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if 0 <= r + dr < rows and 0 <= c + dc < cols
    }

    preset = PRESETS[difficulty]
    walls: set[Coord] = set()
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            if cell in protected or cell in {start, goal}:
                continue
            if rng.random() < preset["wall_rate"]:
                walls.add(cell)

    open_cells = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if (row, col) not in walls and (row, col) not in protected and (row, col) not in {start, goal}
    ]
    rng.shuffle(open_cells)
    enemy_count = min(preset["enemy_count"], max(0, len(open_cells) // 12))
    enemies = set(open_cells[:enemy_count])
    hazard_start = enemy_count
    hazards = set(open_cells[hazard_start : hazard_start + preset["hazard_count"]])

    return WorldState(
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
        walls=walls,
        hazards=hazards,
        enemies=enemies,
        difficulty=difficulty,
        seed=int(seed),
        position=start,
        visited={start},
    )


def _route_for(start: Coord, goal: Coord, rng: random.Random) -> list[Coord]:
    row_delta = -1 if goal[0] < start[0] else 1
    col_delta = -1 if goal[1] < start[1] else 1
    vertical = [row_delta] * abs(start[0] - goal[0])
    horizontal = [col_delta] * abs(start[1] - goal[1])
    steps: list[tuple[str, int]] = []
    while vertical or horizontal:
        choose_horizontal = bool(horizontal) and (not vertical or rng.random() < 0.48)
        steps.append(("col", horizontal.pop()) if choose_horizontal else ("row", vertical.pop()))
    route = [start]
    row, col = start
    for axis, delta in steps:
        if axis == "row":
            row += delta
        else:
            col += delta
        route.append((row, col))
    return route
