"""
More Complex Environment Designs for Quasi-Probability Reinforcement Learning

This file contains specifications for advanced environment designs that would 
more effectively demonstrate the advantages of bidirectional time capabilities
in reinforcement learning.
"""

from qp_rl_project.timeline_grid_env import TimelineGridEnv

def create_deceptive_reward_maze():
    """
    Creates a maze environment with deceptive rewards - areas with high immediate
    rewards that lead to dead ends, while the true optimal path initially has
    lower rewards.
    
    This environment specifically challenges standard RL algorithms because:
    1. Standard Q-learning will be drawn to the high-reward paths
    2. Escaping these deceptive reward traps requires backtracking
    3. Temporal credit assignment is difficult with standard approaches
    
    Returns:
        TimelineGridEnv: A configured environment instance
    """
    # A larger grid for more complex navigation
    grid_size = (10, 10)
    
    # Start and goal positions
    start_pos = (0, 0)
    goal_pos = (9, 9)
    
    # Wall configuration to create a maze with multiple paths
    walls = [
        # Outer boundary walls
        *[(x, 0) for x in range(1, 9)],  # Top boundary
        *[(0, y) for y in range(1, 9)],  # Left boundary
        *[(9, y) for y in range(1, 9)],  # Right boundary
        *[(x, 9) for x in range(1, 9)],  # Bottom boundary
        
        # Inner maze structure - creating multiple pathways
        (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
        # (3, 2), (5, 2), (7, 2), # These were conflicting with one_way_door 'to' positions
        (3, 7), (5, 7), (7, 7),
    ]
    
    # Deceptive reward areas - these give immediate high rewards but lead nowhere
    deceptive_reward_areas = [
        {"position": (3, 3), "reward": 3.0},  # Deceptive reward in first chamber
        {"position": (3, 5), "reward": 4.0},  # Deceptive reward in first chamber
        {"position": (5, 3), "reward": 5.0},  # Deceptive reward in second chamber
        {"position": (5, 5), "reward": 6.0},  # Deceptive reward in second chamber
        {"position": (7, 3), "reward": 7.0},  # Deceptive reward in third chamber
        {"position": (7, 5), "reward": 8.0},  # Deceptive reward in third chamber
    ]
    
    # One-way doors creating traps where you can enter but not exit
    one_way_doors = [
        {'from': (1, 1), 'to': (1, 2), 'action': 1},  # Entry to first chamber
        {'from': (3, 1), 'to': (3, 2), 'action': 1},  # Entry to second chamber
        {'from': (5, 1), 'to': (5, 2), 'action': 1},  # Entry to third chamber
        {'from': (7, 1), 'to': (7, 2), 'action': 1},  # Entry to fourth chamber
    ]
    
    # Strategic portal placements that allow escape from the traps
    portals = {
        (3, 4): (1, 8),  # Escape from first chamber to bottom path
        (5, 4): (3, 8),  # Escape from second chamber to bottom path
        (7, 4): (5, 8),  # Escape from third chamber to bottom path
        (8, 2): (8, 8),  # Shortcut near the end, but hard to reach
    }
    
    return TimelineGridEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        walls=walls,
        one_way_doors=one_way_doors,
        portals=portals,
        custom_rewards=deceptive_reward_areas,
        step_penalty=-0.5  # Slightly lower step penalty to balance with deceptive rewards
    )

def create_dynamic_obstacle_environment():
    """
    Creates an environment where obstacles move in predictable patterns,
    requiring the agent to wait, retreat, or use portals strategically.
    
    This environment specifically challenges standard RL algorithms because:
    1. Optimal solutions require waiting or retreating when obstacles block paths
    2. Temporal patterns require understanding of dynamic state transitions
    3. Premature decisions can lead to unsolvable scenarios
    
    Returns:
        TimelineGridEnv: A configured environment instance with dynamic obstacles
    """
    # Implementation would require extending the TimelineGridEnv to support
    # dynamic obstacles - conceptual design provided here
    
    # Key features to implement:
    # 1. Obstacles that move in fixed patterns (e.g., horizontal or vertical patrols)
    # 2. Scheduled changes to the environment (doors that open/close at specific times)
    # 3. Areas where waiting is optimal (challenging for standard Q-learning)
    # 4. Trap areas where you must jump back in time to escape
    
    # This would demonstrate the power of bidirectional time for solving
    # problems with dynamic elements
    
    grid_size = (7, 7)
    start_pos = (0, 3)
    goal_pos = (6, 3)
    
    walls = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (3, 1), (3, 2), (3, 4), (3, 5), # Opening at (3,3)
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
    ]
    
    # Dynamic obstacles patrol horizontally in the corridors
    # DO1: Patrols row 2, between col 1 and 5. Period 1 per step. Path length 9 (1-5 and back to 1)
    # (2,1)->(2,2)->(2,3)->(2,4)->(2,5)->(2,4)->(2,3)->(2,2)->(2,1)
    do1_path = [(2,1), (2,2), (2,3), (2,4), (2,5), (2,4), (2,3), (2,2)] # Path length 8, not 9
    
    # DO2: Patrols row 4, between col 1 and 5. Period 1 per step.
    # (4,1)->(4,2)->(4,3)->(4,4)->(4,5)->(4,4)->(4,3)->(4,2)->(4,1)
    do2_path = [(4,1), (4,2), (4,3), (4,4), (4,5), (4,4), (4,3), (4,2)]

    dynamic_obstacles_spec = [
        {'path': do1_path, 'period': 1},
        {'path': do2_path, 'period': 1}
    ]
    
    # Custom reward to encourage reaching the middle safe spot
    custom_rewards_spec = [
        {'position': (3,3), 'reward': 0.5} 
    ]

    return TimelineGridEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        walls=walls,
        dynamic_obstacles=dynamic_obstacles_spec,
        custom_rewards=custom_rewards_spec,
        step_penalty=-0.1
    )

def create_sequential_key_environment():
    """
    Creates an environment where the agent must collect keys in a specific
    sequence to unlock doors, but with misleading paths and traps.
    
    This environment specifically challenges standard RL algorithms because:
    1. Long chains of dependencies are difficult for standard RL to learn
    2. Getting keys in wrong order can lead to unrecoverable states
    3. Optimal paths involve collecting keys in the right sequence
    
    Returns:
        TimelineGridEnv: A configured environment instance with keys and doors
    """
    # Implementation would require extending the TimelineGridEnv to support
    # keys and locked doors - conceptual design provided here
    
    # Key features to implement:
    # 1. Multiple keys that must be collected in specific order
    # 2. Doors that only unlock with specific keys
    # 3. False paths that lead to wrong keys first
    # 4. Areas where collecting keys in wrong order creates unrecoverable situations
    #    (without the ability to jump back in time)
    
    # This would showcase how negative probability allows correcting past mistakes
    
    grid_size = (5, 5)
    start_pos = (0, 0)
    goal_pos = (4, 4) # Goal is behind multiple doors
    
    walls = [
        (0, 1), (1, 1), (2, 1), # Wall separating key1 area
        (1, 3), (2, 3), (3, 3), # Wall separating key2 area
    ]
    
    keys_spec = [
        {'id': 'K1', 'position': (1,0)}, # Key 1
        {'id': 'K2', 'position': (3,4)}, # Key 2
    ]
    
    doors_spec = [
        # Door 1 requires K1, at (2,2) to proceed to an intermediate area
        {'position': (2,2), 'key_id_required': 'K1', 'locked': True},
        # Door 2 requires K2, at (4,2) to reach goal area
        {'position': (4,2), 'key_id_required': 'K2', 'locked': True}
    ]
    
    # Custom reward for picking up keys
    custom_rewards_spec = [
        {'position': (1,0), 'reward': 0.2}, # Reward for K1
        {'position': (3,4), 'reward': 0.2}, # Reward for K2
        {'position': (2,2), 'reward': 0.1}, # Small reward for opening D1
    ]

    return TimelineGridEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        walls=walls,
        keys=keys_spec,
        doors=doors_spec,
        custom_rewards=custom_rewards_spec,
        step_penalty=-0.05 # Smaller step penalty
    )

def create_multi_timeline_environment():
    """
    Creates an environment with multiple diverging timelines, where actions in
    one timeline affect possibilities in others.
    
    This environment specifically challenges standard RL algorithms because:
    1. Interdependencies between timelines create intricate causal relationships
    2. Optimal solutions require understanding how one timeline affects others
    3. Jumping between timelines enables solving previously unsolvable problems
    
    This would be the ultimate test for bidirectional time capabilities.
    
    Returns:
        TimelineGridEnv: A configured environment instance with multiple timelines
    """
    # Implementation would require significant extension of TimelineGridEnv
    # to support multiple parallel timelines - conceptual design provided
    
    # Key features to implement:
    # 1. Parallel timelines with different states/obstacles
    # 2. Actions in one timeline affecting states in others
    # 3. Portal-like mechanisms to move between timelines
    # 4. Scenarios that are unsolvable without timeline manipulation
    
    # This would be the most advanced demonstration of bidirectional time
    pass

def create_non_markovian_environment():
    """
    Creates an environment with history-dependent dynamics, where the effect of
    actions depends on the sequence of previous states and actions.
    
    This environment specifically challenges standard RL algorithms because:
    1. The Markov property is violated - past states matter beyond the current state
    2. Standard Q-learning struggles with non-Markovian problems
    3. Negative probability provides a mechanism to revisit and revise history
    
    Returns:
        TimelineGridEnv: A configured non-Markovian environment instance
    """
    # Implementation would require extending TimelineGridEnv to track history
    # and make transition dynamics dependent on that history
    
    # Key features to implement:
    # 1. State transitions that depend on previous sequence of states
    # 2. Rewards that depend on action histories
    # 3. "Memory" grid cells that change behavior based on previous visits
    # 4. Multi-step puzzles where order of operations matters
    
    # This would demonstrate how negative probability helps in problems where
    # history matters beyond the current state representation
    pass

if __name__ == "__main__":
    # Test the deceptive reward maze
    env_deceptive = create_deceptive_reward_maze()
    env_deceptive.reset()
    print("Deceptive Reward Maze Environment:")
    env_deceptive.render()
    
    print("\nThis environment contains:")
    print("- Multiple chambers with high immediate rewards but no way forward")
    print("- One-way doors that create trap situations")
    print("- Portal escape routes that require specific navigation")
    print("- A true optimal path that initially seems less rewarding")
    print("\nStandard RL algorithms will typically get stuck in the high-reward")
    print("chambers, while QP-RL can use backward jumps to escape and find")
    print("the true optimal path.")

    print("\n\n--- Testing Dynamic Obstacle Environment ---")
    env_dynamic = create_dynamic_obstacle_environment()
    obs = env_dynamic.reset()
    print("Initial Dynamic Obstacle Environment:")
    env_dynamic.render()

    # Simulate a few steps to see obstacles move and agent interact
    # Path: (0,3) -> D (1,3) (blocked by wall) -> D (0,3) -> D (0,3) -> D (0,3) -> D (0,3)
    # Try to move Down, then wait by attempting to move into self (e.g. action Up from (0,3))
    # Then move Down once path is clear.
    # Actions: 0:Up, 1:Down, 2:Left, 3:Right

    actions_to_try = [
        1, # Down from (0,3) to (1,3) - hits wall (1,3)
        1, # Down from (0,3) to (1,3) - hits wall (1,3)
        1, # Down from (0,3) to (1,3) - hits wall (1,3)
        1, # Down from (0,3) to (1,3) - hits wall (1,3)
        1, # Down from (0,3) to (1,3) - hits wall (1,3)
        # At this point, DO1 at (2,3) (time=5, path_idx = 5%8 = 5 -> (2,4))
        # Actually, path for DO1 is [(2,1), (2,2), (2,3), (2,4), (2,5), (2,4), (2,3), (2,2)]
        # Time: 0, DO1: (2,1)
        # Time: 1, DO1: (2,2)
        # Time: 2, DO1: (2,3) <- Agent wants to move to (2,3) via (1,3)
        # Time: 3, DO1: (2,4)
        # Time: 4, DO1: (2,5)
        # Time: 5, DO1: (2,4)
    ]
    # Let's redefine a simple scenario: Agent at (0,0), Goal (0,2), DO at (0,1) moves (0,1)->(0,0)->(0,1)
    # This requires a 'wait' action or more complex path planning.
    # The current test in timeline_grid_env.py is more direct for DO functionality.
    # For complex_environment_designs.py, let's just show it can be created.
    
    print("\nSimulating a few steps in Dynamic Obstacle Environment:")
    obs = env_dynamic.reset()
    total_reward_dynamic = 0
    for i in range(10): # Simulate 10 steps
        # Simple agent: always try to move towards goal (Down)
        action_to_take = 1 # Down
        if obs[0] == 2 and obs[1] == 3: # If in corridor before middle opening
             action_to_take = 1 # Down
        elif obs[0] == 4 and obs[1] == 3: # If in corridor after middle opening
             action_to_take = 1 # Down
        
        print(f"Step {i+1}: Obs={obs}, Time={env_dynamic.time_step}, DOs={env_dynamic._current_dynamic_obstacle_positions}")
        print(f"  Taking action: {action_to_take}")
        obs, reward, done, info = env_dynamic.step(action_to_take)
        total_reward_dynamic += reward
        env_dynamic.render()
        print(f"  New Obs={obs}, Reward={reward:.1f}, Done={done}, Info={info}")
        if done:
            print("Goal reached in dynamic environment!")
            break
    print(f"Total reward in dynamic env test: {total_reward_dynamic:.1f}")

    print("\n\n--- Testing Sequential Key Environment ---")
    env_keys = create_sequential_key_environment()
    obs_k = env_keys.reset()
    print("Initial Sequential Key Environment:")
    env_keys.render()

    total_reward_keys = 0
    # Optimal path: Down to (1,0) [K1], Right (1,1)[Wall], Up (0,0), Right (0,1)[Wall]
    # (0,0) -> D (1,0) [K1]
    # (1,0) -> D (2,0)
    # (2,0) -> R (2,1) [Wall] -> R (2,0)
    # (2,0) -> R (2,1) [Wall]
    # (2,0) -> R (2,2) [Door K1] -> Open
    # (2,2) -> D (3,2)
    # (3,2) -> R (3,3) [Wall] -> R (3,4) [K2]
    # Comments detailing path planning and action lists removed for clarity and to avoid syntax errors.
    # The test below uses a simplified action list.
    
    # The environment is created with its default walls.
    # For specific test scenarios, walls can be modified on the instance:
    # env_keys.walls = [(new_wall_config)]
    # obs_k = env_keys.reset() # Important to reset after changing env parameters like walls

    print("Sequential Key Environment (Using default walls from create_sequential_key_environment):")
    env_keys.render() # Renders with initial walls from create_sequential_key_environment

    simple_key_actions = [1, 0, 3, 1, 1, 3, 3, 1, 2, 2, 1, 3, 3] # Approx path, may not be optimal or correct for default walls
    
    print("\nSimulating a few steps in Sequential Key Environment:")
    obs_k = env_keys.reset()
    for i, act in enumerate(simple_key_actions):
        if i > 15 : break # Limit steps for demo
        print(f"Step {i+1}: Obs={obs_k}, Time={env_keys.time_step}, Inv={env_keys.inventory}, Taking action {act}")
        obs_k, reward, done, info = env_keys.step(act)
        total_reward_keys += reward
        env_keys.render()
        print(f"  New Obs={obs_k}, Reward={reward:.1f}, Done={done}, Info={info}")
        if done:
            print("Goal reached in sequential key environment!")
            break
    print(f"Total reward in sequential key env test: {total_reward_keys:.1f}")
