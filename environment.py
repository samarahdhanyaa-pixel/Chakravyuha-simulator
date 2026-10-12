"""Environment transitions and observation generation; policies receive Observation only."""

from __future__ import annotations

from state import DIRECTIONS, Action, Coord, Observation, WorldState
from reward import reward_for
from scenario_generator import PRESETS

VISIBILITY_RADIUS = {"High": 4, "Partial": 3, "Severe": 2}
HAZARD_PRIOR = {"Easy": 0.08, "Medium": 0.14, "Hard": 0.20}


def observe(world: WorldState, visibility: str = "Partial", *, extra_reveal: int = 0) -> Observation:
    """Return a fresh, immutable partial observation with no reference to world truth."""
    if visibility not in VISIBILITY_RADIUS:
        raise ValueError(f"Unknown visibility level: {visibility}")
    radius = VISIBILITY_RADIUS[visibility] + max(0, extra_reveal)
    row, col = world.position
    visible = frozenset(
        (r, c)
        for r in range(world.rows)
        for c in range(world.cols)
        if abs(r - row) + abs(c - col) <= radius
    )
    world.explored.update(visible)
    world.known_walls.update(world.walls & visible)
    world.known_hazards.update(world.hazards & visible)
    world.known_enemies.update(world.enemies & visible)
    return Observation(
        rows=world.rows,
        cols=world.cols,
        position=world.position,
        goal=world.goal,
        visible_cells=frozenset(world.explored),
        visible_walls=frozenset(world.known_walls),
        visible_hazards=frozenset(world.known_hazards),
        visible_enemies=frozenset(world.known_enemies),
        visited_cells=frozenset(world.visited),
        step=world.step,
        difficulty=world.difficulty,
        visibility=visibility,
        status=world.status,
        hazard_prior=HAZARD_PRIOR[world.difficulty],
    )


def step_world(world: WorldState, action: str) -> tuple[float, str]:
    """Apply one action to the true world and return (reward, public event code)."""
    if world.status != "running":
        return 0.0, "terminal"
    world.step += 1

    if action == "SCAN":
        reward = reward_for("scan")
        world.cumulative_reward += reward
        _check_timeout(world)
        return reward, "scan"

    if action not in DIRECTIONS:
        world.invalid_actions += 1
        reward = reward_for("invalid")
        world.cumulative_reward += reward
        _check_timeout(world)
        return reward, "invalid"

    dr, dc = DIRECTIONS[action]
    destination = (world.position[0] + dr, world.position[1] + dc)
    if not (0 <= destination[0] < world.rows and 0 <= destination[1] < world.cols):
        world.invalid_actions += 1
        reward = reward_for("invalid")
        world.cumulative_reward += reward
        _check_timeout(world)
        return reward, "invalid"

    if destination in world.walls:
        world.wall_collisions += 1
        world.visited.add(destination)
        reward = reward_for("wall")
        world.cumulative_reward += reward
        _check_timeout(world)
        return reward, "wall"

    world.position = destination
    world.visited.add(destination)
    if destination in world.enemies:
        world.enemy_collisions += 1
        reward = reward_for("enemy")
        world.cumulative_reward += reward
        world.status = "failure"
        world.termination_reason = "Intercepted by a concealed enemy."
        return reward, "enemy"
    if destination in world.hazards:
        world.hazard_contacts += 1
        reward = reward_for("hazard")
        world.cumulative_reward += reward
        _check_timeout(world)
        return reward, "hazard"
    if destination == world.goal:
        reward = reward_for("goal")
        world.cumulative_reward += reward
        world.status = "success"
        world.termination_reason = "Objective reached."
        return reward, "goal"

    reward = reward_for("move")
    world.cumulative_reward += reward
    _check_timeout(world)
    return reward, "move"


def _check_timeout(world: WorldState) -> None:
    max_steps = PRESETS[world.difficulty]["max_steps"]
    if world.step >= max_steps and world.status == "running":
        world.status = "timeout"
        world.termination_reason = f"Maximum episode length reached ({max_steps} steps)."
