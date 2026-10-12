"""Observation-only action scoring and finite-grid risk-aware path search."""

from __future__ import annotations

import heapq
import random

from state import ARROWS, DIRECTIONS, Decision, Observation


def legal_actions(observation: Observation) -> list[str]:
    """Moves that are in bounds and not blocked by a wall already observed."""
    row, col = observation.position
    actions = []
    for action, (dr, dc) in DIRECTIONS.items():
        destination = (row + dr, col + dc)
        if 0 <= destination[0] < observation.rows and 0 <= destination[1] < observation.cols:
            if destination not in observation.visible_walls:
                actions.append(action)
    return actions


def uncertainty_index(observation: Observation) -> float:
    """Share of battlefield cells that remain unobserved, expressed as an index."""
    total = observation.rows * observation.cols
    return round(100 * (1 - len(observation.visible_cells) / total), 1)


def local_risk_index(observation: Observation, action: str | None = None) -> float:
    """A heuristic exposure index (not a probability of collision)."""
    destination = observation.position
    if action in DIRECTIONS:
        dr, dc = DIRECTIONS[action]
        destination = (destination[0] + dr, destination[1] + dc)
    score = 12.0 if destination not in observation.visible_cells else 5.0
    if destination in observation.visible_hazards:
        score += 36.0
    if destination in observation.visible_enemies:
        score += 70.0
    nearby_unknown = sum(
        1
        for dr, dc in DIRECTIONS.values()
        if (destination[0] + dr, destination[1] + dc) not in observation.visible_cells
        and 0 <= destination[0] + dr < observation.rows
        and 0 <= destination[1] + dc < observation.cols
    )
    score += nearby_unknown * (observation.hazard_prior * 40)
    return round(min(100.0, score), 1)


def _cell_cost(observation: Observation, cell: tuple[int, int]) -> float:
    cost = 1.0
    if cell not in observation.visible_cells:
        # This is a conservative planning surcharge based on the scenario preset,
        # not a claim that a particular cell contains a threat.
        cost += 0.8 + observation.hazard_prior * 5.0
    if cell in observation.visible_hazards:
        cost += 7.0
    if cell in observation.visible_enemies:
        cost += 35.0
    return cost


def estimated_path_cost(observation: Observation, start: tuple[int, int]) -> tuple[float, int]:
    """Weighted Dijkstra using only visible walls and observed or prior costs.

    The grid is finite, and at most rows*cols cells are expanded. Unknown terrain
    remains traversable with a modest prior surcharge until it can be inspected.
    """
    if start == observation.goal:
        return 0.0, 0
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    best = {start: 0.0}
    expanded = 0
    while frontier:
        cost, cell = heapq.heappop(frontier)
        if cost != best.get(cell):
            continue
        expanded += 1
        if cell == observation.goal:
            return cost, expanded
        for dr, dc in DIRECTIONS.values():
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < observation.rows and 0 <= nxt[1] < observation.cols):
                continue
            if nxt in observation.visible_walls:
                continue
            candidate = cost + _cell_cost(observation, nxt)
            if candidate < best.get(nxt, float("inf")):
                best[nxt] = candidate
                heapq.heappush(frontier, (candidate, nxt))
    return float(observation.rows * observation.cols * 2), expanded


def _costs_to_goal(observation: Observation) -> dict[tuple[int, int], float]:
    """One reverse weighted Dijkstra pass gives estimated remaining cost per cell."""
    goal = observation.goal
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, goal)]
    best = {goal: 0.0}
    while frontier:
        cost, cell = heapq.heappop(frontier)
        if cost != best.get(cell):
            continue
        for dr, dc in DIRECTIONS.values():
            predecessor = (cell[0] + dr, cell[1] + dc)
            if not (0 <= predecessor[0] < observation.rows and 0 <= predecessor[1] < observation.cols):
                continue
            if predecessor in observation.visible_walls:
                continue
            # A forward move from predecessor enters the currently expanded cell.
            candidate = cost + _cell_cost(observation, cell)
            if candidate < best.get(predecessor, float("inf")):
                best[predecessor] = candidate
                heapq.heappush(frontier, (candidate, predecessor))
    return best


def action_scores(observation: Observation, strategy: str) -> dict[str, float]:
    actions = legal_actions(observation)
    if not actions:
        return {"SCAN": 0.0}
    remaining_costs = _costs_to_goal(observation) if strategy == "Uncertainty-Aware AI" else {}
    scores: dict[str, float] = {}
    for action in actions:
        dr, dc = DIRECTIONS[action]
        nxt = (observation.position[0] + dr, observation.position[1] + dc)
        if strategy == "Rule-Based":
            # A greedy baseline uses geometric progress and only local known facts.
            old_distance = abs(observation.position[0] - observation.goal[0]) + abs(
                observation.position[1] - observation.goal[1]
            )
            new_distance = abs(nxt[0] - observation.goal[0]) + abs(nxt[1] - observation.goal[1])
            score = 2.0 * (old_distance - new_distance)
            score -= 10.0 if nxt in observation.visible_hazards else 0.0
            score -= 30.0 if nxt in observation.visible_enemies else 0.0
            score -= observation.hazard_prior if nxt not in observation.visible_cells else 0.0
            score += 0.25 if nxt not in observation.visited_cells else 0.0
        elif strategy == "Uncertainty-Aware AI":
            # Penalize observed threats explicitly and favor a small amount of
            # information gain when two routes have similar estimated cost.
            score = -remaining_costs.get(nxt, observation.rows * observation.cols * 2)
            score -= _cell_cost(observation, nxt)
            score -= 5.0 if nxt in observation.visible_hazards else 0.0
            score -= 20.0 if nxt in observation.visible_enemies else 0.0
            score += 0.35 if nxt not in observation.visible_cells else 0.0
        scores[action] = round(score, 3)
    return scores


def build_decision(observation: Observation, strategy: str, rng: random.Random) -> Decision:
    actions = legal_actions(observation)
    if not actions:
        return Decision(
            action="SCAN",
            scores={"SCAN": 0.0},
            risk_index=local_risk_index(observation),
            uncertainty_index=uncertainty_index(observation),
            explanation="No unblocked neighboring move is currently known; scanning expands the observation.",
            known_facts=("All in-bounds neighboring cells are known walls.",),
        )

    if strategy == "Random Agent":
        random_scores = {action: round(rng.random(), 3) for action in actions}
        selected = max(random_scores, key=random_scores.get)
        scores = random_scores
        explanation = "Selected uniformly from moves not known to be blocked; hidden terrain may still cause a collision."
    else:
        scores = action_scores(observation, strategy)
        selected = max(scores, key=scores.get)
        if strategy == "Rule-Based":
            explanation = (
                "The rule-based policy favors Manhattan progress, avoids visible threats, "
                "and slightly rewards an unvisited neighboring cell."
            )
        else:
            explanation = (
                "Weighted Dijkstra compares routes to the objective, adds a preset-based "
                "cost for unseen terrain, and strongly penalizes observed threats."
            )

    unknown_actions = sum(
        1
        for action in actions
        if _destination(observation.position, action) not in observation.visible_cells
    )
    facts = [
        f"{len(actions)} move(s) are not known to be blocked.",
        f"{len(observation.visible_walls)} wall(s), {len(observation.visible_hazards)} hazard(s), "
        f"and {len(observation.visible_enemies)} enemy/enemies are visible.",
        f"{unknown_actions} candidate destination(s) remain unobserved.",
    ]
    return Decision(
        action=selected,
        scores=scores,
        risk_index=local_risk_index(observation, selected),
        uncertainty_index=uncertainty_index(observation),
        explanation=explanation,
        known_facts=tuple(facts),
    )


def _destination(position: tuple[int, int], action: str) -> tuple[int, int]:
    dr, dc = DIRECTIONS[action]
    return position[0] + dr, position[1] + dc


def format_action(action: str) -> str:
    return f"{ARROWS[action]} {action}" if action in ARROWS else action
