#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import time
import random

# Add the parent directory to the path so we can import the minigrid package
sys.path.insert(0, str(Path(__file__).parent.parent))

from minigrid.environment import MiniGridEnv
from minigrid.utils import Actions


def main():
    """
    Example usage of the MiniGridEnv package with negative probabilities.
    """
    print("Initializing MiniGrid Environment...")
    
    # Initialize the environment with a small grid
    env = MiniGridEnv(
        height=8,
        width=8,
        view_range=2,
        wall_density=0.2,
        seed=42  # For reproducibility
    )
    
    # Reset the environment to the initial state
    initial_state = env.reset()
    print("\nInitial Grid:")
    print(env.render(include_hidden=True))
    
    print("\nAgent's Limited Observation (Partial Observability):")
    print(env.render(include_hidden=False))
    
    # Store history of states
    states_history = [initial_state]
    actions_history = []
    
    # Try to navigate to the goal
    max_steps = 20
    step_count = 0
    done = False
    current_state = initial_state
    
    print("\nNavigating through the grid...")
    while not done and step_count < max_steps:
        # Get legal actions
        legal_actions = env.get_legal_actions()
        
        # Choose a smart action - prefer unexplored directions
        if len(legal_actions) == 0:
            print("No legal actions available!")
            break
            
        # Simple heuristic: try to move towards the goal
        # This is a simple approach for the example
        agent_row, agent_col = env.agent_pos
        goal_row, goal_col = env.goal_pos
        
        # Calculate direction to goal
        vertical_diff = goal_row - agent_row
        horizontal_diff = goal_col - agent_col
        
        # Prioritize actions
        action_priority = []
        if vertical_diff < 0:
            action_priority.append(Actions.UP)
        elif vertical_diff > 0:
            action_priority.append(Actions.DOWN)
            
        if horizontal_diff > 0:
            action_priority.append(Actions.RIGHT)
        elif horizontal_diff < 0:
            action_priority.append(Actions.LEFT)
        
        # Add other actions in random order
        remaining_actions = [a for a in legal_actions if a not in action_priority]
        random.shuffle(remaining_actions)
        action_priority.extend(remaining_actions)
        
        # Find the first legal action in our priority list
        action = None
        for a in action_priority:
            if a in legal_actions:
                action = a
                break
                
        if action is None:
            action = random.choice(legal_actions)
        
        # Execute the action
        action_name = Actions.ACTION_NAMES[action]
        print(f"\nStep {step_count + 1}: Taking action {action_name}")
        
        next_state, reward, done, info = env.step(action)
        
        # Update state history
        current_state = next_state
        states_history.append(current_state)
        actions_history.append(action)
        
        # Show the agent's view after the move
        print(env.render(include_hidden=False))
        print(f"Reward: {reward}")
        
        # Check if goal reached
        if done:
            print("Goal reached! 🎉")
        
        step_count += 1
        time.sleep(0.5)  # Brief pause for readability
    
    if not done:
        print("\nFailed to reach the goal within the maximum number of steps.")
    
    # Demonstrate negative probabilities for backward reasoning
    
    print("\n\nDemonstrating Retrospective Inference with Negative Probabilities")
    print("=================================================================")
    
    # Create some hypothetical states with negative probabilities
    # These represent states that might have occurred in alternate histories
    print("\nAdding hypothetical states with negative probabilities...")
    
    # Create a couple of hypothetical previous states
    if len(states_history) >= 3:
        # These would be alternate paths the agent could have taken
        hypothetical_state1 = "hypothetical_1"
        hypothetical_state2 = "hypothetical_2"
        
        # Add negative probability transitions - these represent
        # "might have happened but didn't" trajectories
        env.set_negative_probability(hypothetical_state1, states_history[2], Actions.UP, -0.4)
        env.set_negative_probability(hypothetical_state2, states_history[3], Actions.LEFT, -0.6)
        
        # Now, perform backward reasoning from the final state
        print("\nPerforming backward reasoning from the final state...")
        backward_states = env.backward_reasoning(depth=2)
        
        # Display results
        print("\nResults of backward reasoning:")
        print("---------------------------------")
        for state, probability in backward_states.items():
            if state in states_history:
                idx = states_history.index(state)
                state_desc = f"State after step {idx}"
            elif state == hypothetical_state1:
                state_desc = "Hypothetical state 1 (alternate path)"
            elif state == hypothetical_state2:
                state_desc = "Hypothetical state 2 (alternate path)"
            else:
                state_desc = "Unknown state"
                
            print(f"{state_desc}: probability = {probability:.2f}")
        
        print("\nNote: Negative probabilities allow for retrospective inference")
        print("about states that might have been visited in alternate scenarios.")


if __name__ == "__main__":
    main()
