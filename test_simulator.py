from __future__ import annotations

from dataclasses import fields

import pytest

from agent import STRATEGIES, create_agent
from decision_engine import legal_actions
from environment import observe, step_world
from evaluation import run_episode
from experiments import csv_bytes, run_experiment
from integration import advance, new_simulation
from metrics import summarize_episodes
from reward import reward_for
from scenario_generator import SCENARIOS, generate_scenario
from state import Observation, WorldState


def tiny_world() -> WorldState:
    return WorldState(
        rows=7,
        cols=7,
        start=(3, 3),
        goal=(3, 4),
        walls={(0, 0)},
        hazards={(3, 5)},
        enemies={(4, 3)},
        difficulty="Easy",
        seed=7,
        position=(3, 3),
        visited={(3, 3)},
    )


def test_scenario_generation_is_seeded_and_route_is_playable() -> None:
    first = generate_scenario("Hard", 221)
    second = generate_scenario("Hard", 221)
    assert first.walls == second.walls
    assert first.hazards == second.hazards
    assert first.enemies == second.enemies
    assert first.start == second.start
    assert first.start not in first.walls | first.hazards | first.enemies
    assert first.goal not in first.walls | first.hazards | first.enemies


def test_predefined_scenarios_have_distinct_objectives_and_visibility() -> None:
    objectives = {scenario["goal"] for scenario in SCENARIOS.values()}
    visibilities = {scenario["visibility"] for scenario in SCENARIOS.values()}
    assert len(objectives) == 3
    assert visibilities == {"High", "Partial", "Severe"}
    for scenario in SCENARIOS.values():
        world = generate_scenario(
            scenario["difficulty"],
            scenario["seed"],
            start=scenario["start"],
            goal=scenario["goal"],
        )
        assert world.start == scenario["start"]
        assert world.goal == scenario["goal"]
        assert world.goal not in world.walls | world.hazards | world.enemies


def test_partial_observation_does_not_expose_hidden_truth() -> None:
    world = tiny_world()
    world.hazards = {(6, 6)}
    world.enemies = {(6, 5)}
    observation = observe(world, "Severe")
    assert (0, 0) not in observation.visible_cells
    assert (0, 0) not in observation.visible_walls
    assert (3, 5) not in observation.visible_hazards
    assert (4, 3) not in observation.visible_enemies
    names = {field.name for field in fields(Observation)}
    assert not {"walls", "hazards", "enemies", "world", "true_state"} & names


def test_observations_reveal_and_remember_cells() -> None:
    world = tiny_world()
    world.hazards = {(5, 6)}
    first = observe(world, "Severe")
    world.position = (4, 5)
    second = observe(world, "Severe")
    assert first.visible_cells.issubset(second.visible_cells)
    assert (5, 6) in second.visible_hazards


def test_legal_actions_respect_bounds_and_known_walls() -> None:
    world = tiny_world()
    world.position = (0, 1)
    observation = observe(world, "High")
    actions = legal_actions(observation)
    assert "N" not in actions
    assert "W" not in actions  # (0, 0) is a known wall


def test_goal_detection_and_reward() -> None:
    world = tiny_world()
    reward, event = step_world(world, "E")
    assert event == "goal"
    assert reward == 100
    assert world.status == "success"
    assert world.cumulative_reward == 100
    assert reward_for("move") == -1


def test_wall_hazard_enemy_and_invalid_transitions() -> None:
    world = tiny_world()
    world.walls.add((2, 3))
    wall_reward, wall_event = step_world(world, "N")
    assert wall_event == "wall"
    assert wall_reward == -4
    assert world.position == (3, 3)
    hazard_reward, hazard_event = step_world(world, "E")
    assert hazard_event == "goal"  # goal takes precedence before unrelated cells
    assert hazard_reward == 100

    world = tiny_world()
    world.goal = (0, 6)
    hazard_reward, hazard_event = step_world(world, "S")
    assert hazard_event == "enemy"
    assert hazard_reward == -20
    assert world.status == "failure"

    world = tiny_world()
    invalid_reward, invalid_event = step_world(world, "NOT-A-MOVE")
    assert invalid_event == "invalid"
    assert invalid_reward == -5
    assert world.invalid_actions == 1


def test_hazard_contact_is_penalized_without_terminating() -> None:
    world = tiny_world()
    world.position = (3, 4)
    world.goal = (0, 6)
    reward, event = step_world(world, "E")
    assert event == "hazard"
    assert reward == -7
    assert world.status == "running"
    assert world.hazard_contacts == 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_each_agent_returns_a_candidate_action(strategy: str) -> None:
    world = generate_scenario("Medium", 14)
    observation = observe(world, "Partial")
    decision = create_agent(strategy, 101).decide(observation)
    assert decision.action in set(legal_actions(observation)) | {"SCAN"}
    assert decision.scores


def test_episode_terminates_with_bounded_steps() -> None:
    episode = run_episode("Random Agent", "Easy", "Severe", 131, max_steps=12)
    assert episode["status"] in {"success", "failure", "timeout"}
    assert episode["steps"] <= 12
    assert episode["termination_reason"]


def test_metrics_are_derived_from_episode_results() -> None:
    metrics = summarize_episodes(
        [
            {"status": "success", "steps": 3, "reward": 97, "wall_collisions": 1, "hazard_contacts": 0, "enemy_collisions": 0, "invalid_actions": 0},
            {"status": "timeout", "steps": 4, "reward": -4, "wall_collisions": 0, "hazard_contacts": 1, "enemy_collisions": 0, "invalid_actions": 1},
        ]
    )
    assert metrics["sample_size"] == 2
    assert metrics["success_rate"] == 50
    assert metrics["failure_rate"] == 50
    assert metrics["average_reward"] == 46.5
    assert metrics["average_steps"] == 3.5
    assert metrics["collision_rate"] > 0
    assert metrics["invalid_action_rate"] > 0


def test_experiment_runs_common_seeds_and_exports_csv() -> None:
    raw, summary, config = run_experiment(2, "Easy", "High", STRATEGIES[:2], 50)
    assert len(raw) == 4
    assert len(summary) == 2
    assert config["total_episodes"] == 4
    assert set(raw["seed"]) == {50, 51}
    csv = csv_bytes(raw)
    assert csv.startswith(b"strategy,")
    assert b"termination_reason" in csv


def test_integration_records_replay_and_uses_observation_only_decision() -> None:
    sim = new_simulation(strategy="Uncertainty-Aware AI", difficulty="Easy", visibility="Partial", seed=24)
    assert len(sim["history"]) == 1
    record = advance(sim)
    assert len(sim["history"]) == 2
    assert record["decision"] is not None
    assert record["observation"].visible_cells
    assert "truth" in record
    assert sim["world"].step == 1
