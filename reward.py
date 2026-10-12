"""One authoritative reward function for interactive and batch episodes."""

from __future__ import annotations

from typing import Any


def reward_for(event: str, *, scan: bool = False) -> float:
    """Return the reward attached to an observed transition."""
    if event == "goal":
        return 100.0
    if event == "enemy":
        return -20.0
    if event == "hazard":
        return -7.0
    if event == "wall":
        return -4.0
    if event == "invalid":
        return -5.0
    if event == "scan":
        return -0.5
    return -0.5 if scan else -1.0


def reward_label(reward: float) -> str:
    return f"+{reward:.1f}" if reward > 0 else f"{reward:.1f}"


def reward_breakdown() -> list[dict[str, Any]]:
    return [
        {"Outcome": "Reach the objective", "Reward": "+100"},
        {"Outcome": "Ordinary move", "Reward": "−1"},
        {"Outcome": "Scan for information", "Reward": "−0.5"},
        {"Outcome": "Wall collision", "Reward": "−4"},
        {"Outcome": "Hazard contact", "Reward": "−7"},
        {"Outcome": "Enemy collision", "Reward": "−20"},
    ]
