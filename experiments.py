"""Batch comparison and CSV export over the same simulator used in the UI."""

from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from typing import Any

import pandas as pd

from agent import STRATEGIES
from evaluation import run_episode
from metrics import summarize_episodes


def run_experiment(
    episodes: int = 12,
    difficulty: str = "Medium",
    visibility: str = "Partial",
    strategies: Iterable[str] = STRATEGIES,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare selected policies on exactly the same deterministic seed set."""
    if not 1 <= int(episodes) <= 500:
        raise ValueError("Episode count must be between 1 and 500.")
    chosen = list(dict.fromkeys(strategies))
    if not chosen:
        raise ValueError("Select at least one strategy.")
    unknown = set(chosen) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(sorted(unknown))}")

    records = []
    for offset in range(int(episodes)):
        scenario_seed = int(seed) + offset
        for strategy in chosen:
            records.append(run_episode(strategy, difficulty, visibility, scenario_seed))
    raw = pd.DataFrame.from_records(records)
    metric_rows = []
    for strategy in chosen:
        summary = summarize_episodes(raw.loc[raw["strategy"] == strategy].to_dict("records"))
        metric_rows.append({"strategy": strategy, **summary})
    summary_frame = pd.DataFrame(metric_rows)
    config = {
        "episodes_per_strategy": int(episodes),
        "strategies": chosen,
        "difficulty": difficulty,
        "visibility": visibility,
        "seed_start": int(seed),
        "total_episodes": len(raw),
    }
    return raw, summary_frame, config


def tournament(strategies: Iterable[str], difficulty: str, visibility: str, seed: int, seeds: int = 8) -> pd.DataFrame:
    """Run an actual common-seed tournament; no preset winner is assumed."""
    _, summary, _ = run_experiment(seeds, difficulty, visibility, strategies, seed)
    return summary.sort_values(["success_rate", "average_reward"], ascending=False, kind="stable").reset_index(drop=True)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
