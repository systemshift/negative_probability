"""
Tic-Tac-Toe environment for testing QP-MCTS.
"""

import numpy as np
from typing import List, Tuple, Optional


class TicTacToe:
    """Simple Tic-Tac-Toe game."""

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 or -1

    def copy(self):
        """Create a copy of the game state."""
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """Return list of legal moves (row, col)."""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions

    def make_move(self, action: Tuple[int, int]):
        """Make a move and return new state."""
        row, col = action
        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move: {action}")

        new_game = self.copy()
        new_game.board[row, col] = self.current_player
        new_game.current_player = -self.current_player
        return new_game

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.get_winner() is not None or len(self.get_legal_actions()) == 0

    def get_winner(self) -> Optional[int]:
        """
        Return winner: 1, -1, or None if no winner yet.
        Returns 0 for draw.
        """
        # Check rows
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return int(np.sign(sum(self.board[i, :])))

        # Check columns
        for j in range(3):
            if abs(sum(self.board[:, j])) == 3:
                return int(np.sign(sum(self.board[:, j])))

        # Check diagonals
        if abs(sum(self.board.diagonal())) == 3:
            return int(np.sign(sum(self.board.diagonal())))

        if abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return int(np.sign(sum(np.fliplr(self.board).diagonal())))

        # Check for draw
        if len(self.get_legal_actions()) == 0:
            return 0

        return None

    def get_value(self, player: int = 1) -> float:
        """
        Get value from perspective of player.
        +1 if player wins, -1 if player loses, 0 for draw.
        """
        winner = self.get_winner()
        if winner is None:
            return 0.0
        elif winner == 0:
            return 0.0  # Draw
        elif winner == player:
            return 1.0
        else:
            return -1.0

    def render(self):
        """Print the board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print()
        for i in range(3):
            row = ' '.join(symbols[self.board[i, j]] for j in range(3))
            print(row)
        print()

    def __hash__(self):
        """Hash for storing in dict."""
        return hash((self.board.tobytes(), self.current_player))

    def __eq__(self, other):
        """Equality check."""
        return (np.array_equal(self.board, other.board) and
                self.current_player == other.current_player)


if __name__ == "__main__":
    # Quick test
    print("Testing TicTacToe...")

    game = TicTacToe()
    game.render()

    print("Legal actions:", game.get_legal_actions())

    # Play a few moves
    game = game.make_move((0, 0))  # X
    game.render()

    game = game.make_move((1, 1))  # O
    game.render()

    game = game.make_move((0, 1))  # X
    game.render()

    game = game.make_move((2, 2))  # O
    game.render()

    game = game.make_move((0, 2))  # X wins
    game.render()

    print(f"Terminal: {game.is_terminal()}")
    print(f"Winner: {game.get_winner()}")
    print(f"Value for player 1: {game.get_value(1)}")

    print("\n✓ TicTacToe working!")
