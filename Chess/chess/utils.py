import chess
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Union


def board_to_state(board: chess.Board) -> str:
    """
    Convert a chess board to a hashable state representation.
    
    Args:
        board: A chess.Board object
        
    Returns:
        A string representation of the board state
    """
    return board.fen()


def action_to_uci(action: Tuple[chess.Square, chess.Square, Optional[chess.PieceType]]) -> str:
    """
    Convert an action tuple to UCI string notation.
    
    Args:
        action: A tuple of (from_square, to_square, promotion=None)
        
    Returns:
        The UCI string representation of the move
    """
    from_square, to_square, promotion = action
    
    # Convert squares to algebraic notation
    from_algebraic = chess.square_name(from_square)
    to_algebraic = chess.square_name(to_square)
    
    # Handle promotion
    if promotion:
        promotion_char = chess.piece_symbol(promotion)
        return f"{from_algebraic}{to_algebraic}{promotion_char}"
    else:
        return f"{from_algebraic}{to_algebraic}"


def uci_to_action(uci_move: str) -> Tuple[chess.Square, chess.Square, Optional[chess.PieceType]]:
    """
    Convert a UCI move string to an action tuple.
    
    Args:
        uci_move: Move in UCI notation (e.g., 'e2e4')
        
    Returns:
        Tuple of (from_square, to_square, promotion=None)
    """
    # Parse the UCI move
    from_square = chess.parse_square(uci_move[0:2])
    to_square = chess.parse_square(uci_move[2:4])
    
    # Handle promotion
    promotion = None
    if len(uci_move) > 4:
        promotion_char = uci_move[4]
        promotion = chess.piece_symbol_to_piece_type(promotion_char)
        
    return (from_square, to_square, promotion)


def evaluate_position(board: chess.Board, engine=None) -> float:
    """
    Evaluate a chess position.
    
    If an engine is provided, uses that for evaluation.
    Otherwise returns a simple material-based score.
    
    Args:
        board: A chess.Board object
        engine: Optional chess engine instance
        
    Returns:
        A score from white's perspective
    """
    if engine:
        # If engine is provided, use it for evaluation
        result = engine.analyse(board, chess.engine.Limit(time=0.1))
        score = result['score'].white().score(mate_score=1000)
        return score
    
    # Simple material-based evaluation
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King has no material value
    }
    
    score = 0
    
    # Count material for each side
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
                
    return score


def get_legal_actions(board: chess.Board) -> List[Tuple[chess.Square, chess.Square, Optional[chess.PieceType]]]:
    """
    Get all legal actions from the current board state.
    
    Args:
        board: A chess.Board object
        
    Returns:
        List of legal actions as (from_square, to_square, promotion) tuples
    """
    legal_actions = []
    
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        
        legal_actions.append((from_square, to_square, promotion))
        
    return legal_actions
