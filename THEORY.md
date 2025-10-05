# Quasi-Probability Reinforcement Learning: Theoretical Framework

## Core Concept

Traditional RL agents operate in a **unidirectional time flow**: they can only take actions forward and learn from future rewards. Quasi-Probability RL (QP-RL) introduces **bidirectional temporal reasoning** by allowing Q-values to be negative, representing plausible past actions.

## Theoretical Foundation

### 1. Quasi-Probability Q-Values

In standard Q-learning, Q(s,a) ∈ ℝ⁺ represents expected future return.

In QP-RL, Q(s,a) ∈ ℝ can be:
- **Positive**: Q(s,a) > 0 → "good future action" (standard interpretation)
- **Negative**: Q(s,a) < 0 → "plausible past action" (novel interpretation)
- **Zero**: Q(s,a) = 0 → neutral/unexplored

### 2. Interpretation

The sign of Q(s,a) indicates **temporal direction**:
- |Q(s,a)| = strength/plausibility of the action
- sign(Q(s,a)) = temporal direction (future vs. past)

This creates a **quasi-probability distribution** over actions:
- P(a|s) ∝ |Q(s,a)|  (sample by magnitude)
- Direction determined by sign(Q(s,a))

### 3. Learning Rules

**Forward Update** (for future actions):
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```
Pushes Q(s,a) toward positive values for good actions.

**Backward Update** (for past actions):
```
Q(s_past, a_past) ← Q(s_past, a_past) - β[|Q(s_past, a_past)| - ε]
```
Where:
- β = backward learning rate
- ε = target negative magnitude (small constant)
- This pulls past action Q-values toward -ε

**Key Insight**: Past actions stabilize at small negative values, creating a "memory trace" of plausible trajectories.

### 4. Action Selection

**Sampling from Quasi-Probability Distribution**:

1. For current state s, compute sampling weights: w(a) = |Q(s,a)|
2. Sample action a ~ w(a)
3. Check sign of Q(s,a):
   - If Q(s,a) ≥ 0: Take action a forward (standard RL)
   - If Q(s,a) < 0: **Backward jump** - simulate counterfactual

**Backward Jump Mechanism**:
When Q(s,a) < 0 is sampled:
1. Compute predecessor state s_prev that could have led to s via action a
2. Jump to s_prev (reset environment state)
3. Continue learning from s_prev with updated knowledge

This enables **counterfactual learning**: exploring "what if I had done differently?"

### 5. Counterfactual Reasoning

The backward jumps create a **many-worlds exploration** strategy:
- Agent doesn't just explore forward
- Agent can "rewind" to past states with current knowledge
- This helps escape local optima and explore alternative paths

### 6. Conservation Law

To maintain a valid quasi-probability distribution:
```
Σ_a Q(s,a) = 0  (sum to zero across all actions)
```
This ensures:
- Positive Q-values (future) balanced by negative Q-values (past)
- Distribution normalizes properly for sampling

## Advantages Over Standard RL

1. **Better Exploration**: Can escape local optima by revisiting past states
2. **Counterfactual Learning**: Learn from hypothetical scenarios
3. **Memory of Trajectory**: Negative Q-values encode plausible past paths
4. **Bidirectional Credit Assignment**: Propagate information both forward and backward in time

## Connection to Physics

This framework draws inspiration from:
- **Feynman path integrals**: Sum over all possible paths (including backwards in time)
- **Negative probabilities in quantum mechanics**: Quasi-probability distributions (Wigner functions)
- **Wheeler-Feynman absorber theory**: Time-symmetric interpretation of causality

## Experimental Validation

To validate this approach, we compare:
1. **Classical Q-Learning**: Standard forward-only learning (baseline)
2. **QP-RL**: Full quasi-probability framework with backward jumps

Metrics:
- Convergence speed (episodes to reach goal)
- Sample efficiency (total steps to convergence)
- Exploration quality (state space coverage)
- Final performance (average reward)

Environments where QP-RL should excel:
- Navigation with traps/dead-ends (needs backtracking)
- Sparse reward environments (needs better exploration)
- Multi-path problems (benefits from exploring alternatives)
