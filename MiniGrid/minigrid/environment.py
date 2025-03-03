import numpy as np
import random
from typing import Dict, Tuple, List, Any, Optional, Union

from .markov_tree import MarkovTree
from .utils import (
    Actions, CellType, grid_to_state, state_to_grid, 
    get_observable_area, is_valid_position, get_next_position, render_grid
)


class MiniGridEnv:
    """
    MiniGrid environment implementation with explicit Markov tree and 
    negative probability support for retrospective reasoning about past states.
    
    This environment represents a partially observable gridworld where an agent
    navigates through a grid to reach a goal, with limited visibility.
    """
    
    def __init__(
        self, 
        height: int = 10, 
        width: int = 10, 
        view_range: int = 2,
        wall_density: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize a new MiniGrid environment.
        
        Args:
            height: Grid height
            width: Grid width
            view_range: How far the agent can see (partial observability)
            wall_density: Probability of a cell being a wall
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Initialize grid dimensions
        self.height = height
        self.width = width
        self.view_range = view_range
        self.wall_density = wall_density
        
        # Initialize state variables
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        
        # Initialize Markov tree for state tracking
        self.tree = MarkovTree()
        
        # Track current and previous states
        self.current_state = None
        self.previous_states = []
        
        # Initialize the environment
        self.reset()
    
    def reset(self) -> str:
        """
        Reset the environment to a new random configuration.
        
        Returns:
            The initial state representation
        """
        # Create a new grid filled with empty cells
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Add walls
        self._add_walls()
        
        # Place agent and goal
        self._place_agent_and_goal()
        
        # Reset the Markov tree
        self.tree = MarkovTree()
        
        # Get the initial state
        self.current_state = grid_to_state(self.grid, self.agent_pos)
        self.previous_states = []
        
        # Add the initial state to the tree
        self.tree.add_state(self.current_state)
        
        return self.current_state
    
    def _add_walls(self) -> None:
        """Add walls to the grid based on wall density."""
        # Add border walls
        self.grid[0, :] = CellType.WALL
        self.grid[self.height-1, :] = CellType.WALL
        self.grid[:, 0] = CellType.WALL
        self.grid[:, self.width-1] = CellType.WALL
        
        # Add random walls
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if random.random() < self.wall_density:
                    self.grid[i, j] = CellType.WALL
    
    def _place_agent_and_goal(self) -> None:
        """Place the agent and goal in random, valid positions."""
        # Get all valid positions (not walls)
        valid_positions = []
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if self.grid[i, j] == CellType.EMPTY:
                    valid_positions.append((i, j))
        
        # Ensure we have at least two valid positions
        if len(valid_positions) < 2:
            # Clear some cells if needed
            for i in range(1, self.height-1):
                for j in range(1, self.width-1):
                    if len(valid_positions) < 2:
                        self.grid[i, j] = CellType.EMPTY
                        valid_positions.append((i, j))
        
        # Randomly place agent and goal in distinct positions
        random.shuffle(valid_positions)
        self.agent_pos = valid_positions[0]
        self.goal_pos = valid_positions[1]
        
        # Update grid with goal (agent is tracked separately)
        self.grid[self.goal_pos] = CellType.GOAL
    
    def step(self, action: int) -> Tuple[str, float, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to take (from Actions class)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Store the current state before taking the action
        prev_state = self.current_state
        self.previous_states.append(prev_state)
        
        # Get current position
        row, col = self.agent_pos
        
        # Compute next position
        next_pos = get_next_position(self.agent_pos, action)
        
        # Check if the move is valid
        if is_valid_position(self.grid, next_pos):
            # Update agent position
            self.agent_pos = next_pos
        
        # Check if goal reached
        done = (self.agent_pos == self.goal_pos)
        
        # Get the new state
        next_state = grid_to_state(self.grid, self.agent_pos)
        self.current_state = next_state
        
        # Add the state and transition to the Markov tree
        self.tree.add_state(next_state)
        
        # Add transition with normal probability
        self.tree.add_transition(prev_state, next_state, action, 1.0)
        
        # Calculate reward
        reward = self._calculate_reward(prev_state, next_state, done)
        
        # Additional info
        info = {
            'observable_grid': self.observe(),
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'action': action,
            'action_name': Actions.ACTION_NAMES[action]
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, prev_state: str, next_state: str, done: bool) -> float:
        """
        Calculate the reward for the transition.
        
        Args:
            prev_state: Previous state
            next_state: Current state
            done: Whether the episode is done
            
        Returns:
            Reward value
        """
        # High reward for reaching the goal
        if done:
            return 10.0
            
        # Small penalty for each step to encourage efficiency
        return -0.1
    
    def observe(self) -> np.ndarray:
        """
        Get the agent's observation of the environment (partial observability).
        
        Returns:
            Observable grid with HIDDEN cells where not visible
        """
        return get_observable_area(self.grid, self.agent_pos, self.view_range)
    
    def get_legal_actions(self) -> List[int]:
        """
        Get all legal actions from the current state.
        
        Returns:
            List of legal actions
        """
        legal_actions = []
        
        for action in [Actions.UP, Actions.RIGHT, Actions.DOWN, Actions.LEFT]:
            next_pos = get_next_position(self.agent_pos, action)
            if is_valid_position(self.grid, next_pos):
                legal_actions.append(action)
                
        return legal_actions
    
    def set_negative_probability(self, from_state: str, to_state: str, action: int, probability: float) -> None:
        """
        Set a negative probability for a transition in the Markov tree.
        This can be used for retrospective reasoning and backward inference.
        
        Args:
            from_state: Source state
            to_state: Destination state
            action: The action that causes the transition
            probability: The negative probability value
        """
        # Ensure both states exist
        self.tree.add_state(from_state)
        self.tree.add_state(to_state)
        
        # Add or update the transition with negative probability
        self.tree.add_transition(from_state, to_state, action, probability)
    
    def backward_reasoning(self, depth: int = 1) -> Dict[str, float]:
        """
        Perform backward reasoning from the current state.
        
        This uses negative probabilities to infer previous states.
        
        Args:
            depth: How many steps back to reason
            
        Returns:
            Dictionary mapping potential previous states to their probabilities
        """
        return self.tree.backward_inference(self.current_state, depth)
    
    def get_observation_matrix(self) -> np.ndarray:
        """
        Get a numerical representation of the agent's observation.
        
        Returns:
            Matrix representation suitable for neural networks
        """
        # Get observable grid
        observable = self.observe()
        
        # Create channels for different cell types
        observation = np.zeros((self.height, self.width, 5), dtype=np.float32)
        
        for i in range(self.height):
            for j in range(self.width):
                cell_type = observable[i, j]
                if cell_type != CellType.HIDDEN:
                    observation[i, j, cell_type] = 1.0
                else:
                    # Special channel for hidden cells
                    observation[i, j, 4] = 1.0
                    
        return observation
    
    def render(self, include_hidden: bool = True) -> str:
        """
        Render the environment as a string.
        
        Args:
            include_hidden: Whether to show the full grid or just the observable area
            
        Returns:
            String representation of the environment
        """
        if include_hidden:
            # Render the full grid
            return render_grid(self.grid, self.agent_pos)
        else:
            # Render only the observable area
            observable = self.observe()
            return render_grid(observable, self.agent_pos)
