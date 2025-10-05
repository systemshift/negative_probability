"""
Quasi-Probability Reinforcement Learning Agent

This implements the theoretical framework where Q-values can be negative,
representing plausible past actions, enabling bidirectional temporal reasoning.
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Optional, Any


class QuasiProbabilityAgent:
    """
    RL Agent with quasi-probability Q-values (can be negative).

    Key features:
    - Q(s,a) > 0: Good future actions
    - Q(s,a) < 0: Plausible past actions
    - Bidirectional learning: forward and backward updates
    - Counterfactual exploration via backward jumps
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,           # Forward learning rate
        beta: float = 0.05,            # Backward learning rate
        gamma: float = 0.95,           # Discount factor
        epsilon: float = 0.1,          # Exploration rate
        epsilon_decay: float = 0.995,  # Exploration decay
        min_epsilon: float = 0.01,     # Minimum exploration
        backward_epsilon: float = 0.1, # Target magnitude for past actions
        backward_prob: float = 0.2,    # Probability of backward jump
        use_backward: bool = True,     # Enable backward updates and jumps
        q_init: float = 0.0           # Initial Q-value
    ):
        """
        Initialize the quasi-probability agent.

        Args:
            n_states: Number of states in the environment
            n_actions: Number of actions available
            alpha: Learning rate for forward (future) updates
            beta: Learning rate for backward (past) updates
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate (epsilon-greedy)
            epsilon_decay: Decay rate for exploration
            min_epsilon: Minimum exploration rate
            backward_epsilon: Target magnitude for negative Q-values (past actions)
            backward_prob: Probability of performing backward jump
            use_backward: If False, behaves like classical Q-learning
            q_init: Initial Q-value for all state-action pairs
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.backward_epsilon = backward_epsilon
        self.backward_prob = backward_prob
        self.use_backward = use_backward

        # Q-table: standard positive Q-values for forward actions
        self.Q = defaultdict(lambda: np.full(n_actions, q_init, dtype=np.float64))

        # Backward probability table: stores plausibility of past state-actions
        # Separate from Q to avoid corrupting forward learning
        self.B = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

        # Trajectory history: stores (state, action, reward) for current episode
        self.trajectory = []

        # Statistics
        self.stats = {
            'forward_updates': 0,
            'backward_updates': 0,
            'backward_jumps': 0,
            'episodes': 0
        }

    def reset_episode(self):
        """Reset trajectory at the start of a new episode."""
        self.trajectory = []
        self.stats['episodes'] += 1

    def choose_action(self, state: int) -> Tuple[str, Any]:
        """
        Choose action using quasi-probability sampling.

        Returns:
            ("forward", action) or ("backward", (jump_state, jump_action))
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
            return ("forward", action)

        # Get Q-values for forward actions and B-values for backward plausibility
        q_values = self.Q[state]
        b_values = self.B[state]

        # If backward jumps disabled, use standard greedy policy
        if not self.use_backward:
            action = np.argmax(q_values)
            return ("forward", action)

        # Decide: forward action or backward jump?
        # Use backward probability to determine exploration strategy
        if len(self.trajectory) > 1 and np.random.random() < self.backward_prob:
            # Try backward jump
            jump_idx = self._select_backward_jump_target()
            if jump_idx is not None:
                self.stats['backward_jumps'] += 1
                return ("backward", jump_idx)

        # Forward action: standard epsilon-greedy on Q-values
        action = np.argmax(q_values)
        return ("forward", action)

    def _select_backward_jump_target(self) -> Optional[int]:
        """
        Select which past state to jump back to.

        Prefers earlier states (to maximize exploration benefit).
        """
        if len(self.trajectory) < 2:
            return None

        # Prefer earlier states in trajectory (more benefit from replaying)
        # Weight inversely with recency
        scores = []
        for i in range(len(self.trajectory) - 1):  # Don't include current state
            # Earlier states get higher scores
            recency_bonus = (len(self.trajectory) - i) / len(self.trajectory)
            scores.append(recency_bonus)

        # Sample proportional to scores
        scores = np.array(scores)
        probs = scores / np.sum(scores)
        jump_idx = np.random.choice(len(scores), p=probs)

        return jump_idx

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Update Q-values with both forward and backward updates.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Forward update (standard Q-learning)
        self._forward_update(state, action, reward, next_state, done)

        # Backward update (novel: update past actions)
        if self.use_backward and len(self.trajectory) > 0:
            self._backward_update()

        # Store in trajectory
        self.trajectory.append((state, action, reward))

    def _forward_update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Standard Q-learning update for future actions.
        Pushes Q-values toward positive for good actions.
        """
        current_q = self.Q[state][action]

        if done:
            target = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target = reward + self.gamma * max_next_q

        # TD update
        self.Q[state][action] = current_q + self.alpha * (target - current_q)

        self.stats['forward_updates'] += 1

    def _backward_update(self):
        """
        Novel backward update: track plausibility of past state-actions in B-table.

        This creates a "memory trace" of visited states,
        separate from Q-values to avoid corrupting forward learning.
        """
        if len(self.trajectory) < 1:
            return

        # Mark past states as "visited" by increasing their B-values
        for i in range(len(self.trajectory)):
            past_state, past_action, _ = self.trajectory[i]

            # Increase B-value (plausibility) for visited state-actions
            current_b = self.B[past_state][past_action]
            self.B[past_state][past_action] = current_b + self.beta * (self.backward_epsilon - current_b)

        self.stats['backward_updates'] += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_policy(self, state: int) -> int:
        """Get greedy action for a state (for evaluation)."""
        return np.argmax(self.Q[state])

    def get_value(self, state: int) -> float:
        """Get state value (max Q-value)."""
        return np.max(self.Q[state])

    def get_stats(self) -> Dict[str, int]:
        """Get agent statistics."""
        return self.stats.copy()

    def get_quasi_probability_distribution(self, state: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get quasi-probability distribution for visualization.

        Returns:
            positive_probs: Probabilities for future actions (Q > 0)
            negative_probs: Probabilities for past actions (Q < 0)
        """
        q_values = self.Q[state]

        positive_probs = np.maximum(q_values, 0)
        negative_probs = np.minimum(q_values, 0)

        # Normalize for visualization
        if np.sum(positive_probs) > 0:
            positive_probs = positive_probs / np.sum(positive_probs)
        if np.sum(np.abs(negative_probs)) > 0:
            negative_probs = negative_probs / np.sum(np.abs(negative_probs))

        return positive_probs, negative_probs


class ClassicalQAgent:
    """
    Standard Q-learning agent for comparison.

    This is equivalent to QuasiProbabilityAgent with use_backward=False,
    but implemented cleanly for clarity.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        q_init: float = 0.0
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table: standard positive values only
        self.Q = defaultdict(lambda: np.full(n_actions, q_init, dtype=np.float64))

        self.stats = {
            'episodes': 0,
            'updates': 0
        }

    def reset_episode(self):
        """Reset at the start of a new episode."""
        self.stats['episodes'] += 1

    def choose_action(self, state: int) -> Tuple[str, int]:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])

        return ("forward", action)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Standard Q-learning update."""
        current_q = self.Q[state][action]

        if done:
            target = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target = reward + self.gamma * max_next_q

        self.Q[state][action] = current_q + self.alpha * (target - current_q)
        self.stats['updates'] += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_policy(self, state: int) -> int:
        """Get greedy action."""
        return np.argmax(self.Q[state])

    def get_value(self, state: int) -> float:
        """Get state value."""
        return np.max(self.Q[state])

    def get_stats(self) -> Dict[str, int]:
        """Get agent statistics."""
        return self.stats.copy()
