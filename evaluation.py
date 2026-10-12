"""Reproducible episode execution used by all batch evaluation."""

from __future__ import annotations

import random
from typing import Any

from agent import create_agent
from environment import observe, step_world
from metrics import episode_metrics
from scenario_generator import PRESETS, generate_scenario


def run_episode(
    strategy: str,
    difficulty: str,
    visibility: str,
    seed: int,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run a complete policy episode against a seeded, private world state."""
    world = generate_scenario(difficulty=difficulty, seed=seed)
    policy = create_agent(strategy, seed=seed ^ 0x5A17)
    limit = min(max_steps or PRESETS[difficulty]["max_steps"], PRESETS[difficulty]["max_steps"])
    while world.status == "running" and world.step < limit:
        observation = observe(world, visibility)
        decision = policy.decide(observation)
        step_world(world, decision.action)
    if world.status == "running":
        world.status = "timeout"
        world.termination_reason = f"Evaluation limit reached ({limit} steps)."
    # A stable random stream is reserved for future environment events; current
    # scenarios intentionally use static threats so the same seed is replayable.
    random.Random(seed)
    return episode_metrics(world, strategy=strategy, visibility=visibility)
