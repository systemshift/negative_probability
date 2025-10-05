# Quasi-Probability Reinforcement Learning (QP-RL) v2

A clean, research-grade implementation of reinforcement learning with negative probabilities, enabling bidirectional temporal reasoning and counterfactual learning.

## Overview

Traditional RL agents can only move forward in time. QP-RL introduces **negative Q-values** to represent plausible past actions, enabling:

- **Bidirectional temporal reasoning**: Navigate both forward and backward in time
- **Counterfactual learning**: Explore "what if?" scenarios by jumping to past states
- **Better exploration**: Escape local optima through backward jumps
- **Memory traces**: Negative Q-values encode plausible trajectories

## Theoretical Foundation

See [THEORY.md](../THEORY.md) for full mathematical details.

### Key Concepts

1. **Quasi-Probability Q-Values**: Q(s,a) ∈ ℝ
   - Q(s,a) > 0: Good future action
   - Q(s,a) < 0: Plausible past action
   - sign(Q) indicates temporal direction

2. **Learning Rules**:
   - **Forward update**: Standard TD learning for future actions
   - **Backward update**: Pull past actions toward small negative values

3. **Action Selection**:
   - Sample action proportional to |Q(s,a)|
   - If Q(s,a) < 0: Perform backward jump (counterfactual)
   - If Q(s,a) ≥ 0: Take forward action

## Installation

```bash
# From repository root
pip install -e .
```

## Quick Start

### Run Simple Experiment

```python
from qp_rl_v2 import QuasiProbabilityAgent, ClassicalQAgent, GridWorld
from qp_rl_v2.grid_environment import create_trap_maze
from qp_rl_v2.experiment import compare_agents

# Create environment
env = create_trap_maze()

# Compare agents
qp_results, classical_results = compare_agents(
    env,
    n_episodes=1000,
    n_runs=5,
    verbose=True
)
```

### Run Full Experiment Suite

```bash
# Run on trap maze
python run_experiments.py

# Run on different environment
python run_experiments.py --env long_corridor --episodes 2000

# Run on all environments
python run_experiments.py --all --episodes 1500 --runs 5
```

## Architecture

### Core Components

1. **`quasi_probability_agent.py`**: Main agent implementations
   - `QuasiProbabilityAgent`: Full QP-RL with backward updates
   - `ClassicalQAgent`: Standard Q-learning baseline

2. **`grid_environment.py`**: Test environments
   - `GridWorld`: Configurable grid world
   - `create_trap_maze()`: 5x5 maze with traps
   - `create_long_corridor()`: 3x7 corridor test
   - `create_four_rooms()`: 9x9 classic benchmark

3. **`experiment.py`**: Experiment runner
   - `compare_agents()`: Side-by-side comparison
   - `train_agent()`: Single agent training
   - Result collection and statistics

4. **`visualization.py`**: Visualization utilities
   - Learning curves
   - Policy visualization
   - Quasi-probability heatmaps
   - Comparison bar charts

## Example Results

After running experiments, you'll find:

```
results_v2/
├── trap_maze/
│   ├── learning_curves.png      # Reward/steps over time
│   ├── comparison_bars.png      # Side-by-side metrics
│   ├── policy_qp.png            # QP-RL learned policy
│   ├── policy_classical.png     # Classical policy
│   ├── quasi_probabilities.png  # Positive/negative Q-values
│   └── summary.txt              # Numerical results
```

## Key Hyperparameters

### QuasiProbabilityAgent

- `alpha` (0.1): Forward learning rate
- `beta` (0.05): Backward learning rate
- `gamma` (0.95): Discount factor
- `backward_epsilon` (0.1): Target magnitude for negative Q-values
- `backward_prob` (0.3): Probability of backward jump
- `use_backward` (True): Enable backward updates (set False for classical)

### Environment

- `trap_penalty` (-1.0): Penalty for stepping on trap
- `step_penalty` (-0.01): Penalty per step
- `goal_reward` (1.0): Reward for reaching goal
- `max_steps` (100): Maximum episode length

## Comparison with Classical Q-Learning

The implementation allows direct comparison:

```python
# QP-RL: With backward jumps
qp_agent = QuasiProbabilityAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    use_backward=True  # Enable quasi-probabilities
)

# Classical: Standard Q-learning
classical_agent = ClassicalQAgent(
    n_states=env.n_states,
    n_actions=env.n_actions
)

# Or equivalently:
classical_equiv = QuasiProbabilityAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    use_backward=False  # Disable backward updates
)
```

## When Does QP-RL Help?

QP-RL excels in:

1. **Environments with traps/dead-ends**: Backward jumps escape poor trajectories
2. **Sparse reward tasks**: Better exploration finds rewards faster
3. **Multi-path problems**: Explores alternative routes efficiently

Classical Q-learning may be sufficient for:

1. **Dense reward environments**: Less need for exploration
2. **Simple tasks**: Overhead of backward jumps unnecessary
3. **Small state spaces**: Exhaustive exploration feasible

## Extending the Framework

### Custom Environment

```python
from qp_rl_v2 import GridWorld

env = GridWorld(
    rows=10,
    cols=10,
    start=(0, 0),
    goal=(9, 9),
    walls=[(5, i) for i in range(10)],  # Wall across middle
    traps=[(2, 2), (7, 7)],
    trap_penalty=-2.0,
    step_penalty=-0.01
)
```

### Custom Agent Parameters

```python
agent = QuasiProbabilityAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.2,              # Faster forward learning
    beta=0.1,               # Faster backward learning
    backward_epsilon=0.2,   # Larger negative magnitudes
    backward_prob=0.5,      # More frequent jumps
)
```

## Testing

Run unit tests for each component:

```bash
# Test environment
python qp_rl_v2/grid_environment.py

# Test agent
python qp_rl_v2/quasi_probability_agent.py

# Test experiment runner
python qp_rl_v2/experiment.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quasi_probability_rl,
  title={Quasi-Probability Reinforcement Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/systemshift/negative_probability}
}
```

## License

MIT License - see LICENSE file for details.
