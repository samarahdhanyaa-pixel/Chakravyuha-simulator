"""Shared agent interface and strategy implementations."""

from __future__ import annotations

import random
from typing import Protocol

from decision_engine import build_decision
from state import Decision, Observation

STRATEGIES = ("Random Agent", "Rule-Based", "Uncertainty-Aware AI")


class Agent(Protocol):
    name: str

    def decide(self, observation: Observation) -> Decision: ...


class BaseAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def decide(self, observation: Observation) -> Decision:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    name = "Random Agent"

    def decide(self, observation: Observation) -> Decision:
        return build_decision(observation, self.name, self.rng)


class RuleBasedAgent(BaseAgent):
    name = "Rule-Based"

    def decide(self, observation: Observation) -> Decision:
        return build_decision(observation, self.name, self.rng)


class UncertaintyAwareAgent(BaseAgent):
    name = "Uncertainty-Aware AI"

    def decide(self, observation: Observation) -> Decision:
        return build_decision(observation, self.name, self.rng)


def create_agent(strategy: str, seed: int = 0) -> Agent:
    agent_types = {
        "Random Agent": RandomAgent,
        "Rule-Based": RuleBasedAgent,
        "Uncertainty-Aware AI": UncertaintyAwareAgent,
    }
    try:
        return agent_types[strategy](seed)
    except KeyError as error:
        raise ValueError(f"Unknown strategy: {strategy}") from error
