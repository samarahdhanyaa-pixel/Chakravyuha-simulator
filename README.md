# CHAKRAVYUHA

**An AI-Based Decision-Making Simulator Under Incomplete Information**

> “Every move has a consequence. Every unknown hides a possibility.”

CHAKRAVYUHA is a self-contained, faculty-demo-ready strategy simulator inspired by the layered battlefield formation in the Mahabharata. Agents navigate a seeded battlefield toward a known objective while terrain, hazards, and enemies remain hidden until observed. The app contrasts an agent's partial view with instructor-only ground truth and measures actual outcomes across repeatable scenarios.

## Run the application

The Replit workflow runs the app on port 5173. To start it from a terminal:

```bash
uv run streamlit run app.py --server.address 0.0.0.0 --server.port 5173 --server.headless true
```

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

No API keys, external AI services, database, GPU, or image downloads are required.

## Application tour

- **Command Centre:** seeded battlefield, fog of war, one-step execution, scan action, autoplay to terminal state, decision scores, tactical chronicle, and step-by-step replay.
- **Instructor truth layer:** reveals all walls, hazards, and enemies on the selected replay frame. This display is separate from the `Observation` passed to each agent.
- **Strategy Lab:** inspect the three policy rules and decision audit; run an actual common-seed tournament.
- **Experiment Studio:** select episode count, difficulty, visibility, strategies, and first seed. View computed success, reward, step, failure, collision, and invalid-action metrics; export raw and summary CSV files.
- **Architecture:** see module ownership, information boundaries, implemented algorithms, and justified complexity.
- **Research Notes:** concise problem statement, method, reward schedule, experiment setup, metric definitions, limitations, and a live demonstration sequence.

## Agents and incomplete information

All agents implement the same `Agent` interface and receive an immutable `Observation`; no policy receives a `WorldState`.

1. **Random Agent** chooses from moves that are in bounds and not known to be blocked. Hidden walls can still cause bumps.
2. **Rule-Based** favors Manhattan-distance progress, penalizes visible threats, and slightly favors unvisited cells.
3. **Uncertainty-Aware AI** performs weighted Dijkstra on the finite grid. It excludes known walls, penalizes visible hazards/enemies, and applies an explicit difficulty-based traversal surcharge to unseen cells.

The objective location and grid bounds are known at briefing. Cells are revealed within a Manhattan-radius visibility range and remain explored. A hidden wall causes a non-terminal collision; a hazard subtracts reward but does not end the episode; an enemy collision ends the episode. A scan spends a turn to widen the reveal radius. The risk/exposure display is a heuristic index, not a calibrated probability.

## Reward schedule

| Event | Reward |
| --- | ---: |
| Reach objective | +100 |
| Ordinary move | −1 |
| Information scan | −0.5 |
| Hidden wall collision | −4 |
| Hazard contact | −7 |
| Invalid or out-of-bounds action | −5 |
| Enemy interception | −20 |

Episodes also end at a difficulty-specific step limit.

## Experiments and metrics

Strategies in a comparison are evaluated on the same consecutive integer seeds with a reproducible policy random stream. The app uses the same generator, observations, agents, transitions, and rewards as the live simulator.

- **Success / failure rate:** share of episodes ending in success / failure or timeout.
- **Average cumulative reward:** arithmetic mean of episode reward totals.
- **Average steps:** arithmetic mean of episode action counts.
- **Collision rate:** wall, hazard, and enemy contacts divided by action count.
- **Invalid-action rate:** rejected actions divided by action count.
- **Sample size:** actual episodes per strategy, shown with experiment configuration.

Rates are empirical for the selected sample, not guarantees about future maps. The app does not claim to use neural networks, reinforcement learning, or learned probabilities.

## Module map

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface, command centre, controls, experiment views, and documentation |
| `state.py` | True-state and partial-observation data types |
| `scenario_generator.py` | Seeded difficulty presets, scenarios, threats, and connected protected route |
| `environment.py` | Observation generation, fog of war, movement, collisions, and termination |
| `decision_engine.py` | Legal-action calculation, heuristic indices, and weighted path search |
| `agent.py` | Shared agent interface and three strategies |
| `reward.py` | Single reward definition used by all modes |
| `integration.py` | Episode initialization, step orchestration, and replay snapshots |
| `metrics.py` | Episode and aggregate metric calculations |
| `evaluation.py` | Bounded, reproducible single-episode runner |
| `experiments.py` | Common-seed strategy experiments, tournaments, and CSV generation |
| `visualization.py` | Battlefield rendering, uncertainty radar, and charts |
| `tests/test_simulator.py` | Automated tests for simulation, information boundaries, policies, metrics, and export |

## Tests

```bash
uv run pytest -q
```

The tests cover seeded generation, partial observations and leakage boundaries, remembered discoveries, legal moves, transitions and rewards, all policy interfaces, bounded termination, metric calculations, experiment sample construction, CSV export, and replay integration.

## Limitations

Enemy positions are static. Generated maps use a protected open route so the objective is reachable. The uncertainty cost is a transparent heuristic prior, not a learned belief model or calibrated risk probability. Local comparisons are demonstrations; larger scientific claims require more trials and statistical analysis.
