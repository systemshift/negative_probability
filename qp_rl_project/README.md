# Quasi-Probability Reinforcement Learning for Bidirectional Time

This project implements and evaluates a novel Quasi-Probability Reinforcement Learning (QP-RL) approach that enables bidirectional time dynamics in reinforcement learning agents.

## Overview

The QP-RL framework allows an agent to not only move forward in time (standard RL) but also to selectively "jump backward" to previously visited states when it detects that would be beneficial. This bidirectional capability is achieved through the use of negative probabilities (or quasi-probabilities) which represent "plausibility" of past actions.

### Key Features

1. **Quasi-Probability representation**: Q-values can be positive (future action utility) or negative (past action plausibility)
2. **Temporal bidirectionality**: Agents can move forward in time or jump backward to revisit previous states
3. **Improved exploration**: The backward jumps help escape local optima and explore the state space more efficiently
4. **Comparative framework**: Comparison against standard Q-learning and random restart baselines

## Project Structure

- `qp_agent.py`: Implementation of the core QP-RL agent
- `standard_q_agent.py`: Standard Q-learning agent (baseline 1)
- `random_restart_q_agent.py`: Q-learning with random restarts (baseline 2)
- `timeline_grid_env.py`: Grid environment with portals and one-way doors
- `trap_grid_env.py`: Complex grid environment designed to showcase QP-RL advantages
- `run_experiments.py`: Comparative experimental framework
- `train_qp_agent.py`: Original single-agent training script

## Running the Experiments

To run the comparative experiments on both the standard and trap grid environments:

```bash
python run_experiments.py
```

This will train all agent types on both environments and generate comparative visualizations in the `results/` directory.

## Research Hypothesis

The research aims to demonstrate that negative probabilities can effectively model bidirectional time in reinforcement learning, providing several key advantages:

1. **Faster convergence** to optimal policies in environments with sparse rewards
2. **Better exploration** capabilities, particularly in maze-like environments with potential traps
3. **Escape local optima** more effectively than traditional exploration strategies
4. **Complex decision-making** in environments where actions are partially reversible

## Results Interpretation

The results directory will contain several visualizations:

- Reward comparison plots
- Exploration comparison (unique states visited)
- Jump count comparison
- Q-value heatmaps showing the learned values
- Final statistics for all agent types

When interpreting the results, pay special attention to:

1. How quickly each agent type converges to a good policy
2. How many unique states each agent visits throughout training
3. The relationship between jump counts and performance improvements
4. How effectively agents escape trap areas in the complex environment

## Citation

If you use this code in your research, please cite our work (paper details forthcoming).
