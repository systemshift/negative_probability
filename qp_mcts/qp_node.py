"""
Quasi-Probability Node for MCTS.

Each node tracks both positive probabilities (promising paths)
and negative probabilities (refuted paths).
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple


class QPNode:
    """
    Node in Quasi-Probability MCTS tree.

    Maintains separate Q_positive and Q_negative values to represent:
    - Q_positive: Probability this path leads to good outcome
    - Q_negative: Probability this path has been refuted
    - Q_quasi = Q_positive - Q_negative
    """

    def __init__(self, state, parent=None, action=None, prior=1.0):
        """
        Initialize QP node.

        Args:
            state: Game state at this node
            parent: Parent QPNode (None for root)
            action: Action that led to this node
            prior: Prior probability from policy network (default uniform)
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        # Standard MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[Tuple, 'QPNode'] = {}

        # Quasi-probability components (NEW!)
        self.Q_positive = 0.0    # Evidence this is a good path
        self.Q_negative = 0.0    # Evidence this is a bad path
        self.Q_quasi = 0.0       # Net quasi-probability

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        """Check if game is over at this node."""
        return self.state.is_terminal()

    def expand(self):
        """Expand node by adding children for all legal actions."""
        if self.is_terminal():
            return

        legal_actions = self.state.get_legal_actions()

        # Create child for each legal action
        for action in legal_actions:
            if action not in self.children:
                next_state = self.state.make_move(action)
                prior = 1.0 / len(legal_actions)  # Uniform prior for now
                child = QPNode(next_state, parent=self, action=action, prior=prior)
                self.children[action] = child

    def select_child_ucb(self, c_puct: float = 1.41, use_quasi: bool = True) -> Tuple[Tuple, 'QPNode']:
        """
        Select best child using UCB with quasi-probabilities.

        Args:
            c_puct: Exploration constant
            use_quasi: If True, use Q_quasi; if False, use standard Q

        Returns:
            (action, child) tuple for best child
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            if child.visits == 0:
                # Unvisited node gets infinite score
                score = float('inf')
            else:
                if use_quasi:
                    # Use quasi-probability Q-value
                    q_value = child.Q_quasi
                else:
                    # Standard Q-value
                    q_value = child.value_sum / child.visits

                # UCB exploration term
                u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)

                # Additional term: penalize refuted paths
                refutation_penalty = 0.0
                if use_quasi:
                    refutation_penalty = -0.3 * child.Q_negative

                score = q_value + u_value + refutation_penalty

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def update(self, value: float, alpha: float = 0.1):
        """
        Update node statistics with new value.

        Args:
            value: Value to backpropagate (+1 win, -1 loss, 0 draw)
            alpha: Learning rate for quasi-probability updates
        """
        self.visits += 1
        self.value_sum += value

        # Update quasi-probability components based on outcome
        if value > 0.5:
            # Good outcome - increase positive probability
            self.Q_positive += alpha * (1.0 - self.Q_positive)
        elif value < -0.5:
            # Bad outcome - increase negative probability
            self.Q_negative += alpha * (1.0 - self.Q_negative)
        else:
            # Neutral outcome - small updates
            self.Q_positive += alpha * 0.1 * (0.5 - self.Q_positive)
            self.Q_negative += alpha * 0.1 * (0.5 - self.Q_negative)

        # Compute quasi-probability
        self.Q_quasi = self.Q_positive - self.Q_negative

    def get_average_value(self) -> float:
        """Get average value (standard MCTS)."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def get_visit_counts(self) -> Dict[Tuple, int]:
        """Get visit counts for all children."""
        return {action: child.visits for action, child in self.children.items()}

    def get_best_action(self) -> Optional[Tuple]:
        """Get action with most visits."""
        if not self.children:
            return None
        return max(self.children.items(), key=lambda item: item[1].visits)[0]

    def __repr__(self):
        return (f"QPNode(visits={self.visits}, Q_pos={self.Q_positive:.2f}, "
                f"Q_neg={self.Q_negative:.2f}, Q_quasi={self.Q_quasi:.2f})")


if __name__ == "__main__":
    from tictactoe import TicTacToe

    print("Testing QPNode...")

    game = TicTacToe()
    root = QPNode(game)

    print(f"Root: {root}")
    print(f"Is leaf: {root.is_leaf()}")

    # Expand
    root.expand()
    print(f"After expand: {len(root.children)} children")

    # Update with positive value
    root.update(1.0)
    print(f"After positive update: {root}")

    # Select child
    action, child = root.select_child_ucb()
    print(f"Selected action: {action}")
    print(f"Child: {child}")

    # Update child with negative value
    child.update(-1.0)
    print(f"After negative update: {child}")

    print("\n✓ QPNode working!")
