import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union
import json


# Define action space
class Actions:
    """Action space for the MiniGrid environment."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    # Map for converting actions to direction vectors
    DIRECTION_MAP = {
        UP: (-1, 0),     # Up: move one cell up
        RIGHT: (0, 1),   # Right: move one cell right
        DOWN: (1, 0),    # Down: move one cell down
        LEFT: (0, -1),   # Left: move one cell left
    }
    
    # For string representation
    ACTION_NAMES = {
        UP: "UP",
        RIGHT: "RIGHT",
        DOWN: "DOWN",
        LEFT: "LEFT"
    }


# Cell types for the grid
class CellType:
    """Cell types for the MiniGrid."""
    EMPTY = 0
    WALL = 1
    GOAL = 2
    AGENT = 3
    HIDDEN = 4  # For partially observable environments
    
    # For string representation
    CELL_CHARS = {
        EMPTY: " ",
        WALL: "#",
        GOAL: "G",
        AGENT: "A",
        HIDDEN: "?"
    }


def grid_to_state(grid: np.ndarray, agent_pos: Tuple[int, int]) -> str:
    """
    Convert a grid and agent position to a hashable state representation.
    
    Args:
        grid: 2D numpy array representing the grid
        agent_pos: Tuple of (row, col) representing agent position
        
    Returns:
        A string representation of the state
    """
    # Create a dictionary with the grid and agent position
    state_dict = {
        "grid": grid.tolist(),
        "agent_pos": agent_pos
    }
    
    # Convert to JSON string for hashing
    return json.dumps(state_dict)


def state_to_grid(state: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Convert a state string back to grid and agent position.
    
    Args:
        state: String representation of state
        
    Returns:
        Tuple of (grid, agent_pos)
    """
    state_dict = json.loads(state)
    grid = np.array(state_dict["grid"])
    agent_pos = tuple(state_dict["agent_pos"])
    
    return grid, agent_pos


def get_observable_area(grid: np.ndarray, agent_pos: Tuple[int, int], view_range: int = 2) -> np.ndarray:
    """
    Get the observable area around the agent (for partial observability).
    
    Args:
        grid: Full grid
        agent_pos: Agent position (row, col)
        view_range: How many cells in each direction the agent can see
        
    Returns:
        Observable grid with HIDDEN cells where not visible
    """
    height, width = grid.shape
    row, col = agent_pos
    
    # Create a grid filled with HIDDEN cells
    observable = np.full_like(grid, fill_value=CellType.HIDDEN)
    
    # Fill in the observable area
    for i in range(max(0, row - view_range), min(height, row + view_range + 1)):
        for j in range(max(0, col - view_range), min(width, col + view_range + 1)):
            observable[i, j] = grid[i, j]
    
    return observable


def is_valid_position(grid: np.ndarray, pos: Tuple[int, int]) -> bool:
    """
    Check if a position is valid (within bounds and not a wall).
    
    Args:
        grid: Grid
        pos: Position (row, col)
        
    Returns:
        True if position is valid, False otherwise
    """
    height, width = grid.shape
    row, col = pos
    
    # Check bounds
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    
    # Check if not a wall
    if grid[row, col] == CellType.WALL:
        return False
    
    return True


def get_next_position(pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    """
    Get the next position after taking an action.
    
    Args:
        pos: Current position (row, col)
        action: Action (UP, RIGHT, DOWN, LEFT)
        
    Returns:
        New position (row, col)
    """
    row, col = pos
    drow, dcol = Actions.DIRECTION_MAP[action]
    
    return (row + drow, col + dcol)


def render_grid(grid: np.ndarray, agent_pos: Optional[Tuple[int, int]] = None) -> str:
    """
    Render the grid as a string for display.
    
    Args:
        grid: Grid to render
        agent_pos: Agent position to highlight (if not already in grid)
        
    Returns:
        String representation of the grid
    """
    height, width = grid.shape
    lines = []
    
    # Add top border
    lines.append("+" + "-" * width + "+")
    
    for i in range(height):
        line = "|"
        for j in range(width):
            # If agent_pos is provided and matches current cell, show agent
            if agent_pos and (i, j) == agent_pos:
                line += CellType.CELL_CHARS[CellType.AGENT]
            else:
                cell_type = grid[i, j]
                line += CellType.CELL_CHARS[cell_type]
        line += "|"
        lines.append(line)
    
    # Add bottom border
    lines.append("+" + "-" * width + "+")
    
    return "\n".join(lines)
