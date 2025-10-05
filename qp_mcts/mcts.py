"""
Monte Carlo Tree Search implementations.

Includes both standard MCTS and Quasi-Probability MCTS.
"""

import random
import math
from typing import Optional
from qp_node import QPNode


class MCTS:
    """Standard Monte Carlo Tree Search."""

    def __init__(self, num_simulations: int = 1000, c_puct: float = 1.41):
        """
        Initialize MCTS.

        Args:
            num_simulations: Number of simulations per search
            c_puct: Exploration constant for UCB
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        self.nodes_explored = 0

    def search(self, state) -> QPNode:
        """
        Run MCTS from given state.

        Args:
            state: Initial game state

        Returns:
            Root node with statistics
        """
        self.root = QPNode(state)
        self.nodes_explored = 0

        for _ in range(self.num_simulations):
            # 1. Selection
            node = self._select(self.root)

            # 2. Expansion
            if not node.is_terminal() and node.visits > 0:
                if node.is_leaf():
                    node.expand()
                    self.nodes_explored += len(node.children)

                # Move to one of the children
                if node.children:
                    _, node = node.select_child_ucb(self.c_puct, use_quasi=False)

            # 3. Simulation (Rollout)
            value = self._simulate(node.state)

            # 4. Backpropagation
            self._backpropagate(node, value)

        return self.root

    def _select(self, node: QPNode) -> QPNode:
        """Walk down tree using UCB until reaching leaf."""
        while not node.is_leaf() and not node.is_terminal():
            _, node = node.select_child_ucb(self.c_puct, use_quasi=False)
        return node

    def _simulate(self, state) -> float:
        """
        Random rollout from state to terminal.

        Returns value from perspective of original player at root.
        """
        current_state = state.copy()
        original_player = self.root.state.current_player

        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            action = random.choice(actions)
            current_state = current_state.make_move(action)

        # Get value from perspective of original player
        return current_state.get_value(original_player)

    def _backpropagate(self, node: QPNode, value: float):
        """Update values up to root."""
        current = node

        while current is not None:
            current.update(value, alpha=0.0)  # No quasi-prob updates for standard MCTS
            # Flip value for opponent
            value = -value
            current = current.parent

    def get_best_action(self):
        """Get action with highest visit count."""
        if self.root is None:
            return None
        return self.root.get_best_action()

    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            'nodes_explored': self.nodes_explored,
            'root_visits': self.root.visits if self.root else 0
        }


class QPMCTS(MCTS):
    """
    Quasi-Probability Monte Carlo Tree Search.

    Extends standard MCTS with explicit negative probabilities
    for refuted paths.
    """

    def __init__(self, num_simulations: int = 1000, c_puct: float = 1.41,
                 alpha: float = 0.1):
        """
        Initialize QP-MCTS.

        Args:
            num_simulations: Number of simulations per search
            c_puct: Exploration constant for UCB
            alpha: Learning rate for quasi-probability updates
        """
        super().__init__(num_simulations, c_puct)
        self.alpha = alpha

    def _select(self, node: QPNode) -> QPNode:
        """Walk down tree using QP-UCB until reaching leaf."""
        while not node.is_leaf() and not node.is_terminal():
            _, node = node.select_child_ucb(self.c_puct, use_quasi=True)  # Use quasi!
        return node

    def _backpropagate(self, node: QPNode, value: float):
        """Update values including quasi-probabilities."""
        current = node

        while current is not None:
            # Update with quasi-probability learning
            current.update(value, alpha=self.alpha)
            # Flip value for opponent
            value = -value
            current = current.parent

        # Normalize quasi-probabilities (optional)
        self._normalize_siblings(self.root)

    def _normalize_siblings(self, node: QPNode):
        """
        Normalize quasi-probabilities of children to maintain distribution.

        Ensures sum of |Q_quasi| over siblings is reasonable.
        """
        if not node.children:
            return

        children = list(node.children.values())
        total_abs = sum(abs(child.Q_quasi) for child in children)

        if total_abs > 1.0:
            # Normalize to keep total around 1
            for child in children:
                child.Q_quasi /= total_abs
                # Update components
                if child.Q_quasi >= 0:
                    child.Q_positive = child.Q_quasi
                    child.Q_negative = 0
                else:
                    child.Q_positive = 0
                    child.Q_negative = -child.Q_quasi

        # Recurse to normalize entire tree
        for child in children:
            self._normalize_siblings(child)

    def get_quasi_prob_stats(self) -> dict:
        """Get quasi-probability statistics."""
        if not self.root or not self.root.children:
            return {}

        stats = {}
        for action, child in self.root.children.items():
            stats[action] = {
                'Q_positive': child.Q_positive,
                'Q_negative': child.Q_negative,
                'Q_quasi': child.Q_quasi,
                'visits': child.visits
            }

        return stats


if __name__ == "__main__":
    from tictactoe import TicTacToe

    print("Testing MCTS algorithms...\n")

    # Test Standard MCTS
    print("=" * 60)
    print("Standard MCTS")
    print("=" * 60)

    game = TicTacToe()
    mcts = MCTS(num_simulations=100)

    print("Initial state:")
    game.render()

    root = mcts.search(game)
    best_action = mcts.get_best_action()

    print(f"Best action after 100 simulations: {best_action}")
    print(f"Nodes explored: {mcts.nodes_explored}")
    print(f"Visit counts: {root.get_visit_counts()}")

    # Test QP-MCTS
    print("\n" + "=" * 60)
    print("Quasi-Probability MCTS")
    print("=" * 60)

    game2 = TicTacToe()
    qp_mcts = QPMCTS(num_simulations=100, alpha=0.1)

    print("Initial state:")
    game2.render()

    root2 = qp_mcts.search(game2)
    best_action2 = qp_mcts.get_best_action()

    print(f"Best action after 100 simulations: {best_action2}")
    print(f"Nodes explored: {qp_mcts.nodes_explored}")
    print(f"Visit counts: {root2.get_visit_counts()}")

    print("\nQuasi-probability stats:")
    qp_stats = qp_mcts.get_quasi_prob_stats()
    for action, stats in qp_stats.items():
        print(f"  {action}: Q+={stats['Q_positive']:.2f}, "
              f"Q-={stats['Q_negative']:.2f}, "
              f"Q_quasi={stats['Q_quasi']:.2f}, "
              f"visits={stats['visits']}")

    print("\n✓ Both MCTS variants working!")
