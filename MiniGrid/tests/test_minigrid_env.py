import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the minigrid package
sys.path.insert(0, str(Path(__file__).parent.parent))

from minigrid.environment import MiniGridEnv
from minigrid.markov_tree import MarkovTree
from minigrid.utils import Actions, CellType, grid_to_state, state_to_grid


class TestMarkovTree(unittest.TestCase):
    """Test the MarkovTree class functionality."""
    
    def setUp(self):
        self.tree = MarkovTree()
        
    def test_add_state(self):
        """Test adding states to the tree."""
        self.tree.add_state("state1")
        self.assertTrue("state1" in self.tree.visited_states)
        self.assertTrue("state1" in self.tree.graph.nodes)
        
    def test_add_transition(self):
        """Test adding transitions between states."""
        self.tree.add_transition("state1", "state2", "action1", 1.0)
        self.assertTrue("state1" in self.tree.visited_states)
        self.assertTrue("state2" in self.tree.visited_states)
        self.assertEqual(self.tree.get_probability("state1", "state2"), 1.0)
        
    def test_negative_probability(self):
        """Test setting and getting negative probabilities."""
        self.tree.add_transition("state1", "state2", "action1", -0.5)
        self.assertEqual(self.tree.get_probability("state1", "state2"), -0.5)
        
    def test_backward_inference(self):
        """Test backward inference with negative probabilities."""
        # Create a simple chain of states
        self.tree.add_transition("state1", "state2", "action1", 0.7)
        self.tree.add_transition("state3", "state2", "action2", -0.3)
        
        # Perform backward inference from state2
        prev_states = self.tree.backward_inference("state2", depth=1)
        
        # Check the results
        self.assertEqual(len(prev_states), 2)
        self.assertEqual(prev_states["state1"], 0.7)
        self.assertEqual(prev_states["state3"], -0.3)
        
    def test_multiple_transitions(self):
        """Test handling multiple transitions and multi-step backward inference."""
        # Create a more complex graph
        self.tree.add_transition("state1", "state2", "action1", 0.8)
        self.tree.add_transition("state2", "state3", "action2", 0.6)
        self.tree.add_transition("state4", "state2", "action3", -0.2)
        self.tree.add_transition("state5", "state3", "action4", -0.4)
        
        # Test single-step backward inference
        prev_states = self.tree.backward_inference("state3", depth=1)
        self.assertEqual(len(prev_states), 2)
        self.assertEqual(prev_states["state2"], 0.6)
        self.assertEqual(prev_states["state5"], -0.4)
        
        # Test two-step backward inference
        prev_states = self.tree.backward_inference("state3", depth=2)
        self.assertEqual(len(prev_states), 3)  # state2, state5, state1, state4 (but state1 and state4 combined through state2)
        self.assertEqual(prev_states["state2"], 0.6)
        self.assertEqual(prev_states["state5"], -0.4)
        self.assertEqual(prev_states["state1"], 0.8 * 0.6)  # Combined probability
        self.assertEqual(prev_states["state4"], -0.2 * 0.6)  # Combined probability


class TestMiniGridEnv(unittest.TestCase):
    """Test the MiniGridEnv class functionality."""
    
    def setUp(self):
        # Initialize a small environment with fixed seed for reproducibility
        self.env = MiniGridEnv(height=7, width=7, view_range=2, wall_density=0.1, seed=42)
        
    def test_reset(self):
        """Test resetting the environment."""
        initial_state = self.env.reset()
        
        # Check that a grid and agent position are set
        self.assertIsNotNone(self.env.grid)
        self.assertIsNotNone(self.env.agent_pos)
        self.assertIsNotNone(self.env.goal_pos)
        
        # Check that the state is a string
        self.assertIsInstance(initial_state, str)
        
        # Check that the Markov tree has the initial state
        self.assertIn(initial_state, self.env.tree.visited_states)
        
    def test_step(self):
        """Test making a move in the environment."""
        # Reset the environment
        initial_state = self.env.reset()
        
        # Get legal actions
        legal_actions = self.env.get_legal_actions()
        
        # Ensure there are legal actions
        self.assertTrue(len(legal_actions) > 0)
        
        # Take a step
        action = legal_actions[0]
        next_state, reward, done, info = self.env.step(action)
        
        # Verify the state changed
        self.assertNotEqual(next_state, initial_state)
        
        # Verify the action was recorded in the Markov tree
        self.assertTrue(self.env.tree.graph.has_edge(initial_state, next_state))
        
    def test_observe(self):
        """Test partial observability."""
        self.env.reset()
        
        # Get the full grid and the observable area
        full_grid = self.env.grid.copy()
        observable = self.env.observe()
        
        # Check that the observable area is the same shape as the full grid
        self.assertEqual(observable.shape, full_grid.shape)
        
        # Check that the agent's position is visible
        agent_row, agent_col = self.env.agent_pos
        self.assertNotEqual(observable[agent_row, agent_col], CellType.HIDDEN)
        
        # Check that cells far away from the agent are hidden
        # (assuming view_range is less than grid dimensions)
        hidden_cells = 0
        for i in range(self.env.height):
            for j in range(self.env.width):
                # Skip cells within view range of agent
                if abs(i - agent_row) <= self.env.view_range and abs(j - agent_col) <= self.env.view_range:
                    continue
                    
                if observable[i, j] == CellType.HIDDEN:
                    hidden_cells += 1
                    
        # Ensure there are some hidden cells
        self.assertTrue(hidden_cells > 0)
        
    def test_negative_probability(self):
        """Test setting and using negative probabilities."""
        # Reset the environment
        initial_state = self.env.reset()
        
        # Make a move
        legal_actions = self.env.get_legal_actions()
        action = legal_actions[0]
        next_state, _, _, _ = self.env.step(action)
        
        # Set a negative probability for a hypothetical previous state
        hypothetical_state = "hypothetical_state"
        self.env.set_negative_probability(hypothetical_state, next_state, Actions.UP, -0.3)
        
        # Verify the negative probability was set
        self.assertEqual(
            self.env.tree.get_probability(hypothetical_state, next_state), 
            -0.3
        )
        
    def test_backward_reasoning(self):
        """Test backward reasoning functionality."""
        # Reset the environment
        initial_state = self.env.reset()
        
        # Make several moves
        legal_actions = self.env.get_legal_actions()
        
        # Ensure we have at least 2 legal actions for this test
        if len(legal_actions) >= 2:
            # First move
            action1 = legal_actions[0]
            next_state1, _, _, _ = self.env.step(action1)
            
            # Second move
            legal_actions = self.env.get_legal_actions()
            if len(legal_actions) > 0:
                action2 = legal_actions[0]
                next_state2, _, _, _ = self.env.step(action2)
                
                # Now do backward reasoning
                prev_states = self.env.backward_reasoning(depth=1)
                
                # We should find at least the state after the first move
                self.assertGreater(len(prev_states), 0)
                self.assertEqual(prev_states[next_state1], 1.0)


if __name__ == '__main__':
    unittest.main()
