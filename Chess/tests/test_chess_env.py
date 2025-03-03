import unittest
import chess
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the chess package
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess.environment import ChessEnv
from chess.markov_tree import MarkovTree
from chess.utils import board_to_state, action_to_uci, uci_to_action


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


class TestChessEnv(unittest.TestCase):
    """Test the ChessEnv class functionality."""
    
    def setUp(self):
        # Initialize environment without Stockfish
        self.env = ChessEnv()
        
    def test_reset(self):
        """Test resetting the environment."""
        initial_state = self.env.reset()
        # Check that it's the standard starting position
        self.assertEqual(initial_state, chess.STARTING_FEN)
        
    def test_step(self):
        """Test making a move in the environment."""
        # Reset the environment
        self.env.reset()
        
        # Make a move (e2e4 - King's pawn opening)
        e2 = chess.parse_square("e2")
        e4 = chess.parse_square("e4")
        action = (e2, e4, None)
        
        next_state, reward, done, info = self.env.step(action)
        
        # Verify the board state changed
        self.assertNotEqual(next_state, chess.STARTING_FEN)
        
        # Verify the move was recorded in the Markov tree
        self.assertTrue(self.env.tree.graph.has_edge(chess.STARTING_FEN, next_state))
        
    def test_uci_step(self):
        """Test making a move using UCI string notation."""
        self.env.reset()
        
        # Make a move using UCI string
        next_state, reward, done, info = self.env.step("e2e4")
        
        # Verify the board state changed
        self.assertNotEqual(next_state, chess.STARTING_FEN)
        
    def test_get_legal_actions(self):
        """Test getting legal actions from the current state."""
        self.env.reset()
        
        # Get legal actions from the starting position
        legal_actions = self.env.get_legal_actions()
        
        # There should be 20 legal moves from the starting position
        self.assertEqual(len(legal_actions), 20)
        
    def test_negative_probability(self):
        """Test setting and using negative probabilities."""
        # Reset the environment
        initial_state = self.env.reset()
        
        # Make a move
        e2 = chess.parse_square("e2")
        e4 = chess.parse_square("e4")
        action = (e2, e4, None)
        
        next_state, _, _, _ = self.env.step(action)
        
        # Set a negative probability for a hypothetical previous state
        hypothetical_state = "hypothetical_state"
        self.env.set_negative_probability(hypothetical_state, next_state, "imaginary_action", -0.3)
        
        # Verify the negative probability was set
        self.assertEqual(
            self.env.tree.get_probability(hypothetical_state, next_state), 
            -0.3
        )
        
    def test_backward_reasoning(self):
        """Test backward reasoning functionality."""
        # Reset the environment
        initial_state = self.env.reset()
        
        # Make two moves
        self.env.step("e2e4")  # White's move
        second_state, _, _, _ = self.env.step("e7e5")  # Black's move
        
        # Now do backward reasoning
        prev_states = self.env.backward_reasoning(depth=1)
        
        # We should find the state after White's move
        self.assertTrue(any(prev_state != initial_state for prev_state in prev_states))


if __name__ == '__main__':
    unittest.main()
