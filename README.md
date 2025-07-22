# Negative Probability in Reinforcement Learning

A research implementation exploring the use of negative probabilities (quasi-probabilities) to enable bidirectional temporal navigation in reinforcement learning.

## Overview

This project implements a novel **Temporal Quasi-Probability Reinforcement Learning (TQP-RL)** framework where agents maintain a probability distribution over state-time space with:

- **Current state**: Probability = 1.0 (certainty of being "here and now")
- **Future states**: Positive probabilities in an expanding cone
- **Past states**: Negative probabilities in an expanding cone

The key insight is treating the entire state-space-time as a navigable structure where agents can move both forward and backward in time by sampling from this quasi-probability distribution.

## Core Concept

Traditional RL agents can only move forward in time. Our framework introduces:

1. **Bidirectional Time Navigation**: Agents can navigate to both future states (positive probabilities) and past states (negative probabilities)
2. **Computed Past States**: The system doesn't just remember visited states - it computes plausible past states that could have led to the current state
3. **Quasi-Probability Distribution**: A distribution that sums to 1 but allows negative values, enabling a richer representation of temporal possibilities

## Key Components

### 1. Temporal Navigation Environment (`temporal_nav_env.py`)
A physics-based grid world with:
- State representation: (x, y, vx, vy, t) - position, velocity, and time
- Reversible dynamics through physics laws
- Methods to compute both successor and predecessor states
- Actions apply forces, making the system naturally reversible

### 2. Temporal QP Agent (`temporal_qp_agent.py`)
An agent that:
- Maintains a quasi-probability distribution over state-time space
- Samples from this distribution to choose actions
- Can jump backward to computed past states
- Updates both future and past probabilities through learning

### 3. Original Implementation (for comparison)
- `qp_agent.py`: Original implementation with simpler backward jumping
- `timeline_grid_env.py`: Grid environment with portals and time features
- Standard baselines for comparison

## Installation

```bash
# Clone the repository
git clone https://github.com/systemshift/negative_probability.git
cd negative_probability

# Install the package
pip install -e .
```

## Usage

### Test the Temporal QP Framework

Run the comprehensive temporal quasi-probability experiment:

```bash
python test_temporal_qp.py
```

This will:
1. Create a physics-based environment with obstacles
2. Train both a Temporal QP agent and standard Q-learning agent
3. Generate visualizations of the quasi-probability distribution
4. Compare performance metrics
5. Save results to `results/temporal_qp/`

### Test the Original Implementation

For comparison with the original approach:

```bash
python test_qp_rl.py
```

## Results Interpretation

The temporal QP framework demonstrates several advantages:

1. **Faster Convergence**: By exploring both future and past possibilities
2. **Better Exploration**: The backward jumps help escape local optima
3. **Richer State Representation**: The quasi-probability distribution captures temporal relationships

Key visualizations:
- **Quasi-probability distributions**: Shows the expanding cones of future (positive) and past (negative) probabilities
- **Performance comparison**: Learning curves comparing TQP-RL with standard Q-learning
- **Jump dynamics**: How the agent uses backward time navigation

## Mathematical Foundation

The quasi-probability distribution P(s,t) satisfies:
- ∑ P(s,t) = 1 (normalization)
- P(s_current, t_current) = 1 (certainty)
- P(s, t>t_current) > 0 (future states)
- P(s, t<t_current) < 0 (past states)

This allows the agent to navigate through time by sampling from |P(s,t)| and using the sign to determine direction.

## Future Work

- Extend to continuous state/action spaces
- Apply to more complex environments (Atari, robotics)
- Explore connections to quantum mechanics and Feynman path integrals
- Investigate applications in planning and counterfactual reasoning

## Citation

If you use this code in your research, please cite our work (paper forthcoming).

## License

This project is released under the MIT License.
