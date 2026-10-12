"""Session-friendly orchestration for the environment, agents, rewards, and replay."""

from __future__ import annotations

from typing import Any

from agent import create_agent
from environment import observe, step_world
from scenario_generator import PRESETS, generate_scenario


def new_simulation(
    *,
    strategy: str = "Uncertainty-Aware AI",
    difficulty: str = "Medium",
    visibility: str = "Partial",
    seed: int = 42,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
) -> dict[str, Any]:
    world = generate_scenario(difficulty, seed, start=start, goal=goal)
    simulation: dict[str, Any] = {
        "world": world,
        "strategy": strategy,
        "visibility": visibility,
        "agent": create_agent(strategy, seed ^ 0xA91),
        "reveal_bonus": 0,
        "history": [],
        "latest_decision": None,
        "pending_decision": None,
        "latest_event": "Episode initialized.",
    }
    initial_observation = observe(world, visibility)
    _append_snapshot(simulation, initial_observation, "start", "—", 0.0, None)
    return simulation


def current_observation(simulation: dict[str, Any]):
    return observe(simulation["world"], simulation["visibility"], extra_reveal=simulation["reveal_bonus"])


def pending_decision(simulation: dict[str, Any]):
    if simulation["pending_decision"] is None and simulation["world"].status == "running":
        simulation["pending_decision"] = simulation["agent"].decide(current_observation(simulation))
    return simulation["pending_decision"]


def advance(simulation: dict[str, Any], action: str | None = None) -> dict[str, Any]:
    world = simulation["world"]
    if world.status != "running":
        return simulation["history"][-1]

    current_observation(simulation)
    decision = pending_decision(simulation)
    selected = action or decision.action
    reward, event = step_world(world, selected)
    if selected == "SCAN":
        simulation["reveal_bonus"] = min(5, simulation["reveal_bonus"] + 2)
    updated_observation = current_observation(simulation)
    simulation["latest_decision"] = decision
    simulation["pending_decision"] = None
    simulation["latest_event"] = event
    return _append_snapshot(simulation, updated_observation, event, selected, reward, decision)


def scan(simulation: dict[str, Any]) -> dict[str, Any]:
    return advance(simulation, "SCAN")


def run_to_termination(simulation: dict[str, Any]) -> int:
    """Run only the remaining bounded episode steps; returns actions executed."""
    count = 0
    limit = PRESETS[simulation["world"].difficulty]["max_steps"]
    while simulation["world"].status == "running" and simulation["world"].step < limit:
        advance(simulation)
        count += 1
    return count


def _append_snapshot(
    simulation: dict[str, Any],
    observation: Any,
    event: str,
    action: str,
    reward: float,
    decision: Any,
) -> dict[str, Any]:
    world = simulation["world"]
    snapshot = {
        "step": world.step,
        "observation": observation,
        "position": world.position,
        "status": world.status,
        "termination_reason": world.termination_reason,
        "cumulative_reward": round(world.cumulative_reward, 2),
        "event": event,
        "action": action,
        "reward": reward,
        "decision": decision,
        # Privileged replay truth is stored apart from the immutable observation.
        "truth": {
            "walls": frozenset(world.walls),
            "hazards": frozenset(world.hazards),
            "enemies": frozenset(world.enemies),
        },
    }
    simulation["history"].append(snapshot)
    return snapshot
