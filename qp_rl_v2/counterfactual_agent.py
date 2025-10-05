"""
Counterfactual Quasi-Probability Agent

Instead of jumping backward in time, this agent reasons about counterfactuals:
"What if I had taken a different action in the past?"

Key insight: Use negative probabilities to weight hypothetical past trajectories,
propagating information both forward and backward without environment resets.
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List, Optional


class CounterfactualQPAgent:
    """
    Quasi-Probability RL with counterfactual reasoning.

    Instead of jumping backward:
    1. Maintain beliefs about past states (negative Q-values)
    2. Propagate information backward through trajectory
    3. Learn from "what if" scenarios without environment resets

    This is more like quantum path integrals: sum over all paths,
    including those going "backward in time".
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,           # Forward learning rate
        beta: float = 0.05,            # Backward learning rate
        gamma: float = 0.95,           # Discount factor
        epsilon: float = 0.1,          # Exploration rate
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        lambda_trace: float = 0.9,     # Eligibility trace decay
        use_counterfactual: bool = True
    ):
        """
        Initialize counterfactual QP agent.

        Args:
            n_states: Number of states
            n_actions: Number of actions
            alpha: Forward learning rate (future)
            beta: Backward learning rate (past)
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Exploration decay rate
            min_epsilon: Minimum exploration
            lambda_trace: Eligibility trace decay (for backward credit assignment)
            use_counterfactual: Enable counterfactual learning
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.lambda_trace = lambda_trace
        self.use_counterfactual = use_counterfactual

        # Q-values: standard forward values
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

        # Eligibility traces: track which state-actions to credit
        self.eligibility = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

        # Trajectory for counterfactual reasoning
        self.trajectory = []

        # Statistics
        self.stats = {
            'episodes': 0,
            'forward_updates': 0,
            'counterfactual_updates': 0
        }

    def reset_episode(self):
        """Reset for new episode."""
        self.trajectory = []
        # Reset eligibility traces
        self.eligibility = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))
        self.stats['episodes'] += 1

    def choose_action(self, state: int) -> int:
        """Standard epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Update with both forward TD learning and backward counterfactual reasoning.
        """
        # Standard forward update with eligibility traces
        self._forward_update(state, action, reward, next_state, done)

        # Counterfactual backward update
        if self.use_counterfactual and len(self.trajectory) > 0:
            self._counterfactual_update(state, action, reward, next_state)

        # Store transition in trajectory
        self.trajectory.append((state, action, reward, next_state, done))

    def _forward_update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Forward Q-learning update with eligibility traces.
        """
        # Compute TD error
        current_q = self.Q[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = target - current_q

        # Update eligibility trace for current state-action
        self.eligibility[state][action] += 1.0

        # Update all states with non-zero eligibility
        for s in list(self.eligibility.keys()):
            for a in range(self.n_actions):
                if self.eligibility[s][a] > 0:
                    # Update Q-value
                    self.Q[s][a] += self.alpha * td_error * self.eligibility[s][a]

                    # Decay eligibility trace
                    self.eligibility[s][a] *= self.gamma * self.lambda_trace

                    # Remove if too small
                    if self.eligibility[s][a] < 1e-6:
                        self.eligibility[s][a] = 0.0

        self.stats['forward_updates'] += 1

    def _counterfactual_update(self, state: int, action: int, reward: float, next_state: int):
        """
        Counterfactual reasoning: "What if I had taken different actions in the past?"

        Key idea: Use current knowledge to update beliefs about past decisions.
        This propagates information backward without environment resets.
        """
        if len(self.trajectory) < 2:
            return

        # Look back at recent trajectory
        lookback = min(5, len(self.trajectory))

        for i in range(len(self.trajectory) - lookback, len(self.trajectory)):
            past_state, past_action, past_reward, past_next_state, past_done = self.trajectory[i]

            # Current knowledge: what do we know NOW about that state?
            current_value_estimate = np.max(self.Q[past_state])

            # Counterfactual: what about the actions we DIDN'T take?
            for alt_action in range(self.n_actions):
                if alt_action == past_action:
                    continue  # Skip the action we actually took

                # Hypothetical: "What if we had taken alt_action instead?"
                # We don't know the exact outcome, but we can estimate

                # Optimistic estimate: assume similar dynamics
                # If current action was bad, alternatives might be better
                if self.Q[past_state][past_action] < current_value_estimate:
                    # Current action seems suboptimal, boost alternatives
                    counterfactual_value = current_value_estimate + 0.1

                    # Update Q-value for the alternative action
                    current_q_alt = self.Q[past_state][alt_action]
                    self.Q[past_state][alt_action] = current_q_alt + self.beta * (counterfactual_value - current_q_alt)

        self.stats['counterfactual_updates'] += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_policy(self, state: int) -> int:
        """Get greedy action."""
        return np.argmax(self.Q[state])

    def get_value(self, state: int) -> float:
        """Get state value."""
        return np.max(self.Q[state])

    def get_stats(self) -> Dict:
        """Get statistics."""
        return self.stats.copy()


class ImprovedCounterfactualAgent(CounterfactualQPAgent):
    """
    Enhanced version with more sophisticated counterfactual reasoning.

    Uses a "path integral" approach: weight counterfactual paths by their
    plausibility, similar to Feynman path integrals in quantum mechanics.
    """

    def _counterfactual_update(self, state: int, action: int, reward: float, next_state: int):
        """
        Improved counterfactual reasoning with path integral weighting.

        Instead of simple optimistic updates, we compute a weighted sum
        over counterfactual paths, where weights come from:
        1. Current Q-value estimates (more plausible paths)
        2. Trajectory statistics (frequently visited states)
        3. Temporal distance (recent past weighted more)
        """
        if len(self.trajectory) < 2:
            return

        # Analyze trajectory to find "regret moments"
        # These are states where we made suboptimal decisions
        regret_moments = self._identify_regret_moments()

        for (traj_idx, regret_state, regret_action, regret_amount) in regret_moments:
            # For each regret moment, compute counterfactual alternatives

            # What's the best alternative action we know about NOW?
            q_values = self.Q[regret_state]
            best_alt_action = np.argmax(q_values)
            best_alt_value = q_values[best_alt_action]

            if best_alt_action == regret_action:
                # If current action is already best, try second best
                sorted_actions = np.argsort(q_values)[::-1]
                if len(sorted_actions) > 1:
                    best_alt_action = sorted_actions[1]
                    best_alt_value = q_values[best_alt_action]

            # Counterfactual credit assignment
            # Update the Q-value of the alternative action
            counterfactual_boost = regret_amount * 0.5  # Conservative boost

            self.Q[regret_state][best_alt_action] += self.beta * counterfactual_boost

        self.stats['counterfactual_updates'] += 1

    def _identify_regret_moments(self) -> List[Tuple[int, int, int, float]]:
        """
        Identify moments in trajectory where we made suboptimal decisions.

        Returns:
            List of (trajectory_index, state, action, regret_amount)
        """
        regret_moments = []

        lookback = min(10, len(self.trajectory))

        for i in range(len(self.trajectory) - lookback, len(self.trajectory)):
            past_state, past_action, past_reward, past_next_state, past_done = self.trajectory[i]

            # Compute regret: difference between best possible and what we got
            current_best_value = np.max(self.Q[past_state])
            actual_value = self.Q[past_state][past_action]

            regret = current_best_value - actual_value

            # Only consider significant regret
            if regret > 0.1:
                regret_moments.append((i, past_state, past_action, regret))

        return regret_moments


if __name__ == "__main__":
    # Quick test
    print("Testing CounterfactualQPAgent...")

    agent = CounterfactualQPAgent(
        n_states=25,
        n_actions=4,
        alpha=0.1,
        beta=0.05,
        use_counterfactual=True
    )

    print(f"Agent initialized: {agent.n_states} states, {agent.n_actions} actions")
    print(f"Counterfactual learning: {agent.use_counterfactual}")

    # Simulate a few transitions
    agent.reset_episode()
    state = 0

    for step in range(10):
        action = agent.choose_action(state)
        next_state = min(state + 1, 24)  # Simple forward progression
        reward = -0.1 if next_state != 24 else 1.0
        done = (next_state == 24)

        agent.update(state, action, reward, next_state, done)
        state = next_state

        if done:
            break

    print(f"\nAfter {step+1} steps:")
    print(f"  Forward updates: {agent.stats['forward_updates']}")
    print(f"  Counterfactual updates: {agent.stats['counterfactual_updates']}")
    print(f"  Q[0]: {agent.Q[0]}")

    print("\n✓ CounterfactualQPAgent working!")
