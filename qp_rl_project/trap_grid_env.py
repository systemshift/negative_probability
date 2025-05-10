from qp_rl_project.timeline_grid_env import TimelineGridEnv

def create_trap_grid():
    """
    Creates a more challenging grid environment with trap areas and complex navigation.
    
    This environment is specifically designed to test the benefits of bidirectional time
    mechanics in RL by featuring:
    1. Multiple "trap" areas that are easy to enter but difficult to exit
    2. Narrow passages that make exploration challenging
    3. One-way doors that could lead to dead ends
    4. Strategic portal placements that can help escape traps but require planning
    
    Returns:
        TimelineGridEnv: A configured environment instance
    """
    # A larger grid for more complex navigation
    grid_size = (8, 8)
    
    # Start and goal positions
    start_pos = (0, 0)
    goal_pos = (7, 7)
    
    # Wall configuration to create a maze-like structure
    walls = [
        # Outer boundaries to force a specific path
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        
        # Inner walls creating narrow passages
        (2, 2), (2, 3), (2, 4), (2, 5),  # Removed (2,6) as it conflicts with portal
        (3, 2), (4, 2), (5, 2), (6, 2),
        (4, 4), (4, 5), (5, 4),
        
        # Dead-end traps
        (1, 7), (2, 7), (3, 7),
        (7, 1), (7, 2), (7, 3)
    ]
    
    # One-way doors leading to challenging situations
    one_way_doors = [
        # Door leading to a "trap" area that's hard to escape from
        {'from': (1, 5), 'to': (1, 6), 'action': 3},  # Right from (1,5) to (1,6)
        
        # Door leading to a shortcut
        {'from': (3, 3), 'to': (5, 3), 'action': 1},  # Down from (3,3) to (5,3)
        
        # Door that blocks a key path
        {'from': (6, 4), 'to': (6, 5), 'action': 3},  # Right from (6,4) to (6,5)
    ]
    
    # Strategic portal placements
    portals = {
        # Potential rescue from trap area
        (2, 6): (4, 1),
        
        # Potential shortcut if accessible
        (5, 6): (6, 6),
        
        # Risky portal that could lead to a trap or closer to goal
        (3, 1): (6, 3)
    }
    
    return TimelineGridEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        walls=walls,
        one_way_doors=one_way_doors,
        portals=portals
    )

if __name__ == "__main__":
    # Test the trap grid environment
    env = create_trap_grid()
    env.reset()
    print("Trap Grid Environment:")
    env.render()
    
    print("\nTesting a sample path through the environment...")
    actions = [
        1, 3, 3, 3, 3, 3, 3, 1,  # Navigate initial corridors
        1, 1, 2, 1, 1, 3, 3, 3   # Try to reach goal
    ]
    
    state = env.reset()
    total_reward = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: Action {action}, New State {next_state}, Reward {reward}")
        env.render()
        
        if done:
            print(f"Goal reached! Total reward: {total_reward}")
            break
        
        state = next_state
    
    if not done:
        print(f"Goal not reached after sample path. Total reward so far: {total_reward}")
