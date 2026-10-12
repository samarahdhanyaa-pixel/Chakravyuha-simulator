"""Metric calculations shared by single episodes, tournaments, and experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def summarize_episodes(episodes: Iterable[dict[str, Any]]) -> dict[str, float]:
    records = list(episodes)
    sample_size = len(records)
    if sample_size == 0:
        return {
            "sample_size": 0,
            "success_rate": 0.0,
            "average_reward": 0.0,
            "average_steps": 0.0,
            "failure_rate": 0.0,
            "collision_rate": 0.0,
            "invalid_action_rate": 0.0,
        }
    successful = sum(episode["status"] == "success" for episode in records)
    failures = sum(episode["status"] in {"failure", "timeout"} for episode in records)
    total_actions = sum(max(1, episode["steps"]) for episode in records)
    collisions = sum(episode.get("wall_collisions", 0) + episode.get("hazard_contacts", 0) + episode.get("enemy_collisions", 0) for episode in records)
    invalid = sum(episode.get("invalid_actions", 0) for episode in records)
    return {
        "sample_size": float(sample_size),
        "success_rate": 100.0 * successful / sample_size,
        "average_reward": float(np.mean([episode["reward"] for episode in records])),
        "average_steps": float(np.mean([episode["steps"] for episode in records])),
        "failure_rate": 100.0 * failures / sample_size,
        "collision_rate": 100.0 * collisions / total_actions,
        "invalid_action_rate": 100.0 * invalid / total_actions,
    }


def episode_metrics(world: Any, *, strategy: str, visibility: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "visibility": visibility,
        "difficulty": world.difficulty,
        "seed": world.seed,
        "status": world.status,
        "steps": world.step,
        "reward": round(world.cumulative_reward, 2),
        "wall_collisions": world.wall_collisions,
        "hazard_contacts": world.hazard_contacts,
        "enemy_collisions": world.enemy_collisions,
        "invalid_actions": world.invalid_actions,
        "termination_reason": world.termination_reason,
    }
