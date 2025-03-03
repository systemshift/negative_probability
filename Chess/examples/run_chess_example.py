#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import time

# Add the parent directory to the path so we can import the chess package
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess.environment import ChessEnv
from chess.markov_tree import MarkovTree


def main():
    """
    Example usage of the ChessEnv package with negative probabilities.
    """
    print("Initializing Chess Environment...")
    
    # Initialize the environment
    # You can optionally provide the path to a Stockfish binary
    # env = ChessEnv(stockfish_path="/path/to/stockfish")
    env = ChessEnv()
    
    # Reset the environment to the initial state
    initial_state = env.reset()
    print("\nInitial Board Position:")
    print(env.render())
    
    # Make some moves
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    
    current_state = initial_state
    states_history = [current_state]
    
    print("\nPlaying a sequence of moves...")
    for i, move in enumerate(moves):
        print(f"\nMove {i+1}: {move}")
        next_state, reward, done, info = env.step(move)
        current_state = next_state
        states_history.append(current_state)
        
        print(env.render())
        print(f"Reward: {reward:.2f}")
        print(f"Current player: {info['turn']}")
        
        if info['is_check']:
            print("Check!")
        if done:
            print("Game over!")
            break
        
        time.sleep(0.5)  # Pause briefly between moves
    
    # Demonstrate negative probabilities for backward reasoning
    
    print("\n\nDemonstrating Retrospective Inference with Negative Probabilities")
    print("=================================================================")
    
    # Add some hypothetical states with negative probabilities
    # These represent states that might have happened but didn't
    hypothetical_state1 = "hypothetical_position_1"
    hypothetical_state2 = "hypothetical_position_2"
    
    # Add negative probability transitions
    print("\nAdding hypothetical states with negative probabilities...")
    env.set_negative_probability(hypothetical_state1, states_history[2], "imaginary_action1", -0.3)
    env.set_negative_probability(hypothetical_state2, states_history[3], "imaginary_action2", -0.7)
    
    # Perform backward reasoning from the current state
    print("\nPerforming backward reasoning from current state...")
    backward_states = env.backward_reasoning(depth=2)
    
    # Display results
    print("\nResults of backward reasoning:")
    print("---------------------------------")
    for state, probability in backward_states.items():
        if state in states_history:
            idx = states_history.index(state)
            state_desc = f"State after move {idx}" if idx > 0 else "Initial state"
        elif state == hypothetical_state1:
            state_desc = "Hypothetical state 1"
        elif state == hypothetical_state2:
            state_desc = "Hypothetical state 2"
        else:
            state_desc = "Unknown state"
            
        print(f"{state_desc}: probability = {probability:.2f}")
    
    print("\nNote: Negative probabilities allow for retrospective inference about states")
    print("that might have occurred in alternate paths or interpretations of history.")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
