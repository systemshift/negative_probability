"""
Simple Grid World Environment for Testing Quasi-Probability RL

Designed to showcase when backward jumps and counterfactual learning help.
"""

import numpy as np
from typing import Tuple, List, Optional


class GridWorld:
    """
    Grid world with configurable obstacles, traps, and rewards.

    State space: Each cell is a state (indexed 0 to rows*cols-1)
    Action space: 0=Up, 1=Down, 2=Left, 3=Right
    """

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (4, 4),
        walls: Optional[List[Tuple[int, int]]] = None,
        traps: Optional[List[Tuple[int, int]]] = None,
        trap_penalty: float = -1.0,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        max_steps: int = 100
    ):
        """
        Initialize grid world.

        Args:
            rows: Number of rows
            cols: Number of columns
            start: Starting position (row, col)
            goal: Goal position (row, col)
            walls: List of wall positions (impassable)
            traps: List of trap positions (large penalty, but can escape)
            trap_penalty: Reward for stepping on trap
            step_penalty: Penalty for each step
            goal_reward: Reward for reaching goal
            max_steps: Maximum steps per episode
        """
        self.rows = rows
        self.cols = cols
        self.start_pos = start
        self.goal_pos = goal
        self.walls = set(walls) if walls else set()
        self.traps = set(traps) if traps else set()
        self.trap_penalty = trap_penalty
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        # State/action space
        self.n_states = rows * cols
        self.n_actions = 4  # Up, Down, Left, Right

        # Current state
        self.agent_pos = start
        self.steps = 0

        # For visualization
        self.visit_count = np.zeros((rows, cols), dtype=int)

    def reset(self) -> int:
        """Reset environment to start state."""
        self.agent_pos = self.start_pos
        self.steps = 0
        self.visit_count = np.zeros((self.rows, self.cols), dtype=int)
        self.visit_count[self.agent_pos] += 1
        return self._pos_to_state(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take action in environment.

        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode is done
            info: Additional information
        """
        # Get next position
        next_pos = self._get_next_pos(self.agent_pos, action)

        # Check if hit wall (stay in place)
        if next_pos in self.walls:
            next_pos = self.agent_pos

        # Update position
        self.agent_pos = next_pos
        self.steps += 1
        self.visit_count[self.agent_pos] += 1

        # Calculate reward
        reward = self.step_penalty

        if self.agent_pos in self.traps:
            reward = self.trap_penalty

        done = False
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
        elif self.steps >= self.max_steps:
            done = True

        next_state = self._pos_to_state(self.agent_pos)

        info = {
            'position': self.agent_pos,
            'steps': self.steps
        }

        return next_state, reward, done, info

    def set_state(self, state: int):
        """
        Manually set environment state (for backward jumps).

        Args:
            state: State index to set
        """
        self.agent_pos = self._state_to_pos(state)
        # Note: we don't reset steps, as this is a "counterfactual" jump

    def _get_next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next position after taking action."""
        row, col = pos

        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.rows - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.cols - 1, col + 1)

        return (row, col)

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert position to state index."""
        row, col = pos
        return row * self.cols + col

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to position."""
        row = state // self.cols
        col = state % self.cols
        return (row, col)

    def render(self, mode: str = 'human'):
        """Render the grid world."""
        if mode == 'human':
            print(f"\nStep {self.steps}/{self.max_steps}")
            for row in range(self.rows):
                line = ""
                for col in range(self.cols):
                    pos = (row, col)
                    if pos == self.agent_pos:
                        line += " A "
                    elif pos == self.goal_pos:
                        line += " G "
                    elif pos in self.walls:
                        line += " # "
                    elif pos in self.traps:
                        line += " X "
                    else:
                        line += " . "
                print(line)
            print()

    def render_visits(self):
        """Render visit counts (for analyzing exploration)."""
        print("\nVisit counts:")
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                count = self.visit_count[row, col]
                line += f"{count:3d} "
            print(line)
        print()


def create_trap_maze() -> GridWorld:
    """
    Create a maze with traps - showcases benefit of backward jumps.

    Layout (5x5):
    S . . . .
    . X # X .
    . X # X .
    . . # . .
    . . . . G

    S = Start, G = Goal, # = Wall, X = Trap
    Agent must navigate around traps and walls.
    """
    walls = [(1, 2), (2, 2), (3, 2)]
    traps = [(1, 1), (1, 3), (2, 1), (2, 3)]

    return GridWorld(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        walls=walls,
        traps=traps,
        trap_penalty=-1.0,
        step_penalty=-0.01,
        goal_reward=1.0,
        max_steps=100
    )


def create_long_corridor() -> GridWorld:
    """
    Create a long corridor with traps - tests exploration.

    Layout (3x7):
    S . . X . . .
    # # # # # # G
    . . . X . . .

    Agent must explore to find the path around the wall.
    """
    walls = [(1, i) for i in range(6)]  # Middle row is wall except goal
    traps = [(0, 3), (2, 3)]

    return GridWorld(
        rows=3,
        cols=7,
        start=(0, 0),
        goal=(1, 6),
        walls=walls,
        traps=traps,
        trap_penalty=-0.5,
        step_penalty=-0.01,
        goal_reward=1.0,
        max_steps=50
    )


def create_four_rooms() -> GridWorld:
    """
    Create classic four-rooms environment.

    Layout (9x9):
    S . . . # . . . .
    . . . . # . . . .
    . . . . . . . . .
    . . . . # . . . .
    # # . # # # . # #
    . . . . # . . . .
    . . . . . . . . .
    . . . . # . . . .
    . . . . # . . . G

    Tests exploration across rooms.
    """
    walls = []
    # Vertical walls
    walls.extend([(i, 4) for i in range(9) if i != 2 and i != 6])
    # Horizontal walls
    walls.extend([(4, i) for i in range(9) if i != 2 and i != 6])

    return GridWorld(
        rows=9,
        cols=9,
        start=(0, 0),
        goal=(8, 8),
        walls=walls,
        traps=[],
        step_penalty=-0.01,
        goal_reward=1.0,
        max_steps=200
    )


if __name__ == "__main__":
    # Test environments
    print("=== Testing Trap Maze ===")
    env = create_trap_maze()
    env.render()

    print("\n=== Testing Long Corridor ===")
    env2 = create_long_corridor()
    env2.render()

    print("\n=== Testing Four Rooms ===")
    env3 = create_four_rooms()
    env3.render()

    # Test basic interaction
    print("\n=== Testing Basic Interaction ===")
    env = create_trap_maze()
    state = env.reset()
    print(f"Start state: {state}")
    env.render()

    # Take some actions
    actions = [1, 1, 3, 3, 1, 1, 3, 3]  # Try to reach goal
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.3f}, Done: {done}")
        env.render()
        if done:
            break

    env.render_visits()
