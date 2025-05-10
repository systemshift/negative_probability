# Negative Probability in Reinforcement Learning

A research implementation of Quasi-Probability Reinforcement Learning (QP-RL) that uses negative probabilities to enable bidirectional time dynamics in RL.

## Overview

This project demonstrates how negative probabilities can open up new possibilities in reinforcement learning by allowing agents to selectively "jump backward" in time to previously visited states. The framework enables agents to:

1. Move forward in time (standard RL)
2. Jump backward to revisit promising previous states
3. Escape local optima more effectively
4. Achieve faster convergence in complex environments

## Concept

Negative (or quasi) probabilities represent an extension to standard probability theory that has found applications in quantum mechanics and other fields. In our reinforcement learning context:

- Positive Q-values represent "utility of future actions" (standard RL)
- Negative Q-values represent "plausibility of past actions"
- When a negative Q-value is sampled during action selection, the agent may jump back to a previous state

This bidirectional time capability provides a novel approach to the exploration-exploitation trade-off in reinforcement learning.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/negative_probability.git
cd negative_probability

# Install the package
pip install -e .
```

## Project Structure

- **qp_rl_project/**: Main package directory
  - `qp_agent.py`: Core implementation of the QP-RL agent
  - `standard_q_agent.py`: Standard Q-learning agent (baseline)
  - `random_restart_q_agent.py`: Q-learning with random restarts (baseline)
  - `timeline_grid_env.py`: Grid environment with time manipulation features
  - `trap_grid_env.py`: Complex grid environment designed to showcase QP-RL benefits
  - `run_experiments.py`: Comprehensive experimental framework
  - `train_qp_agent.py`: Original training script for single agents

## Usage

### Quick Test

Run a quick test comparing the QP-RL agent to a standard Q-learning agent:

```bash
python test_qp_rl.py
```

### Comprehensive Experiments

Run comprehensive experiments with all agent types on different environments:

```bash
python -m qp_rl_project.run_experiments
```

This will:
1. Train all agent types on both standard and trap grid environments
2. Generate comparative visualizations including reward plots, exploration metrics, and Q-value heatmaps
3. Save results in the `results/` directory organized by environment type

## Results

The most interesting comparison metrics include:

1. **Reward progression**: How quickly each agent type converges to optimal policies
2. **Exploration efficiency**: How many unique states each agent visits during training
3. **Jump dynamics**: How the QP-RL agent utilizes backward jumps over time
4. **Escape from traps**: How effectively agents handle the trap areas in complex environments

## Research Hypothesis

The main research hypothesis is that negative probabilities can effectively model bidirectional time in reinforcement learning, providing several key advantages:

1. **Faster convergence** to optimal policies in environments with sparse rewards
2. **Better exploration** capabilities in maze-like environments with potential traps
3. **Escape local optima** more effectively than traditional exploration strategies
4. **Complex decision-making** in environments where actions are partially reversible

## Extending the Framework

You can extend this framework by:

1. Creating new environment types
2. Implementing more advanced QP-RL variants
3. Adding different baselines for comparison
4. Testing with continuous state/action spaces

## License

This project is released under the MIT License.

## Citation

If you use this code in your research, please cite our work (paper details forthcoming).
