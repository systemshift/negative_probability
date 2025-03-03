import chess
import chess.engine
import numpy as np
import os
import random
from typing import Tuple, Dict, List, Optional, Any, Union
from pathlib import Path

from .markov_tree import MarkovTree
from .utils import board_to_state, action_to_uci, uci_to_action, evaluate_position, get_legal_actions


class ChessEnv:
    """
    Chess environment implementation that explicitly represents states and actions
    as a Markov tree, supporting negative probabilities for retrospective inference.
    """
    
    def __init__(self, stockfish_path: Optional[str] = None):
        """
        Initialize a new chess environment.
        
        Args:
            stockfish_path: Optional path to Stockfish binary for evaluation
        """
        # Initialize the chess board
        self.board = chess.Board()
        
        # Initialize the Markov tree for state tracking
        self.tree = MarkovTree()
        
        # Initialize Stockfish engine if path is provided
        self.engine = None
        if stockfish_path and os.path.exists(stockfish_path):
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            
        # Track current and previous states
        self.current_state = None
        self.previous_states = []
        
        # Initialize the environment
        self.reset()
    
    def reset(self, fen: Optional[str] = None) -> str:
        """
        Reset the chess environment to the starting position or a specified FEN.
        
        Args:
            fen: Optional FEN string to set the board position
            
        Returns:
            The initial state representation
        """
        # Reset the board
        if fen:
            self.board.set_fen(fen)
        else:
            self.board.reset()
            
        # Reset the Markov tree
        self.tree = MarkovTree()
        
        # Get the initial state
        self.current_state = board_to_state(self.board)
        self.previous_states = []
        
        # Add the initial state to the tree
        self.tree.add_state(self.current_state)
        
        return self.current_state
    
    def step(self, action: Union[Tuple[chess.Square, chess.Square, Optional[chess.PieceType]], str]) -> Tuple[str, float, bool, Dict]:
        """
        Execute a move in the chess environment.
        
        Args:
            action: Either a tuple (from_square, to_square, promotion) or UCI string
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert string action to tuple if needed
        if isinstance(action, str):
            action = uci_to_action(action)
            
        from_square, to_square, promotion = action
        
        # Create the move object
        move = chess.Move(from_square, to_square, promotion)
        
        # Verify the move is legal
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move}")
            
        # Store the current state before making the move
        prev_state = self.current_state
        self.previous_states.append(prev_state)
        
        # Make the move
        self.board.push(move)
        
        # Get the new state
        next_state = board_to_state(self.board)
        self.current_state = next_state
        
        # Add the state and transition to the Markov tree
        self.tree.add_state(next_state)
        self.tree.add_transition(prev_state, next_state, action, 1.0)  # Normal probability
        
        # Check if the game is over
        done = self.board.is_game_over()
        
        # Calculate reward
        reward = self._calculate_reward(prev_state, next_state)
        
        # Additional info
        info = {
            'legal_actions': get_legal_actions(self.board),
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'fullmove_number': self.board.fullmove_number,
            'turn': 'white' if self.board.turn == chess.WHITE else 'black',
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, prev_state: str, next_state: str) -> float:
        """
        Calculate the reward for the transition from prev_state to next_state.
        
        Args:
            prev_state: Previous board state
            next_state: Current board state
            
        Returns:
            Reward value
        """
        # Evaluate the position
        position_value = evaluate_position(self.board, self.engine)
        
        # Base reward on position evaluation
        reward = position_value / 10.0  # Scale down the evaluation
        
        # Additional rewards for game outcomes
        if self.board.is_checkmate():
            # Large reward/penalty for checkmate
            reward = 10.0 if self.board.turn == chess.BLACK else -10.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            # Small penalty for draw
            reward = -0.1
            
        return reward
    
    def get_legal_actions(self) -> List[Tuple[chess.Square, chess.Square, Optional[chess.PieceType]]]:
        """
        Get all legal actions from the current state.
        
        Returns:
            List of legal actions as (from_square, to_square, promotion) tuples
        """
        return get_legal_actions(self.board)
    
    def set_negative_probability(self, from_state: str, to_state: str, action, probability: float) -> None:
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
    
    def get_observation(self) -> np.ndarray:
        """
        Get a numerical representation of the current board state.
        
        Returns:
            8x8x12 numpy array representing the board
        """
        # Initialize an empty board representation
        observation = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Map each piece type and color to a channel
        pieces = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP, 
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        
        # Fill the observation array
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Get the square's rank and file (0-7)
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Determine which channel to use based on piece type and color
                piece_idx = pieces.index(piece.piece_type)
                if piece.color == chess.BLACK:
                    piece_idx += 6  # Offset for black pieces
                
                # Set the value to 1.0 at the piece's position
                observation[rank, file, piece_idx] = 1.0
        
        return observation
    
    def render(self) -> str:
        """
        Render the current board state as a string.
        
        Returns:
            String representation of the board
        """
        return str(self.board)
    
    def close(self) -> None:
        """
        Clean up resources.
        """
        if self.engine:
            self.engine.quit()
