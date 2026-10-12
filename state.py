"""Typed state shared by the Chakravyuha simulator modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Coord = tuple[int, int]
Action = Literal["N", "E", "S", "W", "SCAN"]

DIRECTIONS: dict[str, Coord] = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
ARROWS: dict[str, str] = {"N": "↑", "E": "→", "S": "↓", "W": "←"}


@dataclass
class WorldState:
    """Evaluator/environment-only truth. Never pass this object to an agent."""

    rows: int
    cols: int
    start: Coord
    goal: Coord
    walls: set[Coord]
    hazards: set[Coord]
    enemies: set[Coord]
    difficulty: str
    seed: int
    position: Coord
    step: int = 0
    cumulative_reward: float = 0.0
    visited: set[Coord] = field(default_factory=set)
    explored: set[Coord] = field(default_factory=set)
    known_walls: set[Coord] = field(default_factory=set)
    known_hazards: set[Coord] = field(default_factory=set)
    known_enemies: set[Coord] = field(default_factory=set)
    status: str = "running"
    termination_reason: str = ""
    invalid_actions: int = 0
    wall_collisions: int = 0
    hazard_contacts: int = 0
    enemy_collisions: int = 0


@dataclass(frozen=True)
class Observation:
    """Information available to a policy at a single decision point."""

    rows: int
    cols: int
    position: Coord
    goal: Coord
    visible_cells: frozenset[Coord]
    visible_walls: frozenset[Coord]
    visible_hazards: frozenset[Coord]
    visible_enemies: frozenset[Coord]
    visited_cells: frozenset[Coord]
    step: int
    difficulty: str
    visibility: str
    status: str
    hazard_prior: float


@dataclass(frozen=True)
class Decision:
    action: str
    scores: dict[str, float]
    risk_index: float
    uncertainty_index: float
    explanation: str
    known_facts: tuple[str, ...]
