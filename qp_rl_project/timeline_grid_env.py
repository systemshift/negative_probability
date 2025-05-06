import collections

class TimelineGridEnv:
    def __init__(self, grid_size=(5,5), start_pos=(0,0), goal_pos=None, walls=None, one_way_doors=None, portals=None):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        
        if start_pos[0] < 0 or start_pos[0] >= self.rows or \
           start_pos[1] < 0 or start_pos[1] >= self.cols:
            raise ValueError(f"Start position {start_pos} is outside the grid boundaries.")
        self.start_pos = start_pos

        if goal_pos is None:
            self.goal_pos = (self.rows - 1, self.cols - 1)
        else:
            if goal_pos[0] < 0 or goal_pos[0] >= self.rows or \
               goal_pos[1] < 0 or goal_pos[1] >= self.cols:
                raise ValueError(f"Goal position {goal_pos} is outside the grid boundaries.")
            self.goal_pos = goal_pos
        
        self.walls = walls if walls is not None else []
        # Ensure walls are within bounds and not on start/goal
        for wall in self.walls:
            if not (0 <= wall[0] < self.rows and 0 <= wall[1] < self.cols):
                raise ValueError(f"Wall {wall} is outside grid boundaries.")
            if wall == start_pos:
                raise ValueError(f"Wall {wall} cannot be at the start position.")
            if wall == self.goal_pos:
                raise ValueError(f"Wall {wall} cannot be at the goal position.")

        # Actions must be defined before validating one_way_doors and other components that might use them
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        self.action_space_size = len(self.actions)

        self.one_way_doors = one_way_doors if one_way_doors is not None else []
        # Example one_way_doors: [{'from': (r1,c1), 'to': (r2,c2), 'action': action_idx}]
        # Basic validation for one-way doors
        for door in self.one_way_doors:
            if not all(k in door for k in ['from', 'to', 'action']):
                raise ValueError(f"One-way door {door} is missing 'from', 'to', or 'action' key.")
            for pos_key in ['from', 'to']:
                pos = door[pos_key]
                if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                    raise ValueError(f"Position {pos} in one-way door {door} is outside grid boundaries.")
            if door['action'] not in self.actions:
                 raise ValueError(f"Action {door['action']} in one-way door {door} is invalid.")
            if door['from'] in self.walls or door['to'] in self.walls:
                raise ValueError(f"One-way door {door} cannot involve a wall position.")

        self.portals = portals if portals is not None else {}
        # Example portals: {(entry_r, entry_c): (exit_r, exit_c)}
        # Basic validation for portals
        for entry_pos, exit_pos in self.portals.items():
            for pos_label, pos in [("entry", entry_pos), ("exit", exit_pos)]:
                if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                    raise ValueError(f"Portal {pos_label} position {pos} is outside grid boundaries.")
                if pos in self.walls:
                     raise ValueError(f"Portal {pos_label} position {pos} cannot be on a wall.")
            if entry_pos == self.goal_pos:
                raise ValueError(f"Portal entry {entry_pos} cannot be at the goal position.")
            # It's okay for a portal exit to be the goal.

        self.agent_pos = None # Will be set in reset()

    def reset(self):
        """
        Resets the agent to the starting position.
        Returns the initial observation (agent's position).
        """
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action_idx):
        """
        Performs one step in the environment.
        Args:
            action_idx (int): The index of the action to take.
        Returns:
            tuple: (observation, reward, done, info)
                   observation (tuple): Agent's new position (row, col).
                   reward (float): Reward received after taking the action.
                   done (bool): Whether the episode has ended.
                   info (dict): Auxiliary diagnostic information (empty for now).
        """
        if action_idx not in self.actions:
            raise ValueError(f"Invalid action index: {action_idx}. Must be in {list(self.actions.keys())}")

        dr, dc = self.actions[action_idx]
        
        current_row, current_col = self.agent_pos
        
        # Check for one-way door transition
        one_way_door_taken = False
        for door in self.one_way_doors:
            if self.agent_pos == door['from'] and action_idx == door['action']:
                next_row, next_col = door['to']
                one_way_door_taken = True
                # print(f"DEBUG: Took one-way door from {door['from']} to {door['to']} with action {action_idx}")
                break
        
        if not one_way_door_taken:
            # Standard move if no one-way door was applicable for this action from this state
            next_row = current_row + dr
            next_col = current_col + dc

            # Check for wall collision for standard moves
            if (next_row, next_col) in self.walls:
                next_row, next_col = current_row, current_col # Stay in place
                reward = -1.0 # Penalty for hitting a wall
                done = False
                self.agent_pos = (next_row, next_col)
                return self.agent_pos, reward, done, {}

            # Boundary checks for standard moves
            if next_row < 0 or next_row >= self.rows:
                next_row = current_row # Stay in place
            if next_col < 0 or next_col >= self.cols:
                next_col = current_col # Stay in place
        
        self.agent_pos = (next_row, next_col)

        # Check for portal transition after movement
        if self.agent_pos in self.portals:
            # print(f"DEBUG: Agent at {self.agent_pos}, which is a portal entry.")
            self.agent_pos = self.portals[self.agent_pos]
            # print(f"DEBUG: Agent teleported to {self.agent_pos}.")
            # Reward for taking a portal can be neutral or specific. Keeping it part of the step cost for now.

        # Reward logic
        reward = -0.1 # Small penalty for each step
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 1.0 # Reward for reaching the goal
            done = True
        
        return self.agent_pos, reward, done, {}

    def render(self, mode='human'):
        """
        Renders the environment.
        For 'human' mode, prints to console.
        """
        if mode == 'human':
            for r in range(self.rows):
                row_str = ""
                for c in range(self.cols):
                    pos = (r,c)
                    if pos == self.agent_pos:
                        row_str += "A "
                    elif pos == self.goal_pos:
                        row_str += "G "
                    elif pos in self.walls:
                        row_str += "# "
                    elif pos in self.portals: # Mark portal entry
                        row_str += "P "
                    elif pos in self.portals.values() and pos not in self.portals: # Mark portal exit if not also an entry
                         # This check is a bit tricky if an exit can be an entry to another portal
                         # For simplicity, just marking entries for now.
                         # A more robust way would be to have a list of all portal cells.
                        is_also_an_entry = any(val_pos == pos for val_pos in self.portals.keys())
                        if not is_also_an_entry: # only mark as 'O' if it's purely an exit
                            row_str += "O " # 'O' for Out/Exit of a portal
                        else: # If an exit is also an entry for another portal, it will be marked 'P'
                            row_str += ". " 
                    else:
                        row_str += ". "
                print(row_str)
            print("-" * (self.cols * 2)) # Separator
        else:
            # Potentially return a more structured representation for other modes
            pass

if __name__ == '__main__':
    # Example Usage
    print("Testing basic TimelineGridEnv (Phase 1)...")
    env_basic = TimelineGridEnv(grid_size=(3,4), start_pos=(0,0), goal_pos=(2,3))
    obs = env_basic.reset()
    env_basic.render()

    done = False
    total_reward = 0
    max_steps = 20
    
    # Basic Env Test (Phase 1) - Commented out for brevity in later phases
    # for step_num in range(max_steps):
    #     if done:
    #         break
    #     action = env_basic.action_space_size -1 
    #     if obs == (0,3): action = 1 
    #     if obs == (1,3): action = 1 
    #     print(f"Step {step_num + 1}: Taking action {action}")
    #     obs, reward, done, info = env_basic.step(action)
    #     total_reward += reward
    #     env_basic.render()
    #     print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
    # print(f"Episode finished after {step_num + 1} steps. Total reward: {total_reward}")


    print("\n--- Testing TimelineGridEnv with Walls (Phase 2) ---")
    wall_list_p2 = [(1,1), (1,2), (2,1)]
    env_walls = TimelineGridEnv(grid_size=(4,4), start_pos=(0,0), goal_pos=(3,3), walls=wall_list_p2)
    obs_w = env_walls.reset()
    # env_walls.render() # Rendered during action loop
    total_reward_w = 0

    # Test navigation with walls
    # actions_to_take_w = [
    #     3, # Right to (0,1) from (0,0)
    #     1, # Down from (0,1) to (1,1) - should hit wall at (1,1), stay at (0,1)
    #     0, # Up from (0,1) - hit boundary, stay at (0,1)
    #     3, # Right from (0,1) to (0,2)
    #     1, # Down from (0,2) to (1,2) - should hit wall at (1,2), stay at (0,2)
    #     3, # Right from (0,2) to (0,3)
    #     1, # Down from (0,3) to (1,3)
    #     1, # Down from (1,3) to (2,3)
    #     2, # Left from (2,3) to (2,2)
    #     1, # Down from (2,2) to (3,2)
    #     3, # Right from (3,2) to (3,3) - Goal!
    # ]
    # env_walls.reset()
    # for i, action in enumerate(actions_to_take_w):
    #     print(f"Step {i+1}: Current obs: {obs_w}, Taking action {action}")
    #     obs_w, reward, done, _ = env_walls.step(action)
    #     total_reward_w += reward
    #     env_walls.render()
    #     print(f"New Obs: {obs_w}, Reward: {reward:.1f}, Done: {done}")
    #     if done:
    #         print("Goal reached!")
    #         break
    # print(f"Total reward with walls: {total_reward_w:.1f}")

    # Test invalid wall positions - Commented out for brevity
    # try:
    #     TimelineGridEnv(grid_size=(3,3), walls=[(10,10)])
    # except ValueError as e:
    #     print(f"\nCaught expected error for wall out of bounds: {e}")
    # try:
    #     TimelineGridEnv(grid_size=(3,3), start_pos=(1,1), walls=[(1,1)])
    # except ValueError as e:
    #     print(f"Caught expected error for wall on start: {e}")
    # try:
    #     TimelineGridEnv(grid_size=(3,3), goal_pos=(2,2), walls=[(2,2)])
    # except ValueError as e:
    #     print(f"Caught expected error for wall on goal: {e}")

    print("\n--- Testing TimelineGridEnv with One-Way Doors (Phase 3) ---")
    one_way_door_list_p3 = [
        {'from': (1,2), 'to': (2,2), 'action': 1} 
    ]
    env_owd = TimelineGridEnv(grid_size=(4,4), start_pos=(0,1), goal_pos=(2,3), 
                              walls=[(1,1)], one_way_doors=one_way_door_list_p3)
    obs_owd = env_owd.reset()
    # env_owd.render() # Rendered during action loop
    total_reward_owd = 0
    
    # actions_to_take_owd = [
    #     3, # R: (0,1) -> (0,2)
    #     1, # D: (0,2) -> (1,2) (now at 'from' of one-way door)
    #     1, # D: (1,2) -> (2,2) (takes one-way door)
    #     3, # R: (2,2) -> (2,3) (Goal)
    # ]
    # print("Path to take one-way door:")
    # for i, action in enumerate(actions_to_take_owd):
    #     print(f"Step {i+1}: Current obs: {obs_owd}, Taking action {action}")
    #     obs_owd, reward, done, _ = env_owd.step(action)
    #     total_reward_owd += reward
    #     env_owd.render()
    #     print(f"New Obs: {obs_owd}, Reward: {reward:.1f}, Done: {done}")
    #     if done:
    #         print("Goal reached via one-way door!")
    #         break
    # print(f"Total reward (one-way door path): {total_reward_owd:.1f}")
    
    # Test invalid one_way_door configs - Commented out
    # try:
    #     TimelineGridEnv(grid_size=(3,3), one_way_doors=[{'from':(0,0), 'to':(20,20), 'action':0}])
    # except ValueError as e:
    #     print(f"\nCaught expected error for OWD out of bounds: {e}")


    print("\n--- Testing TimelineGridEnv with Portals (Phase 4) ---")
    # Grid:
    # S . . .
    # . # . P  (P at (1,3) is portal entry to (3,1))
    # . . . .
    # . O . G  (O at (3,1) is portal exit, G at (3,3) is goal)
    portal_map_p4 = {(1,3): (3,1)}
    env_portal = TimelineGridEnv(grid_size=(4,4), start_pos=(0,0), goal_pos=(3,3),
                                 walls=[(1,1)], portals=portal_map_p4)
    obs_p = env_portal.reset()
    env_portal.render()
    total_reward_p = 0

    # Corrected path for portal test:
    # Start: (0,0), Wall: (1,1), Portal Entry: (1,3), Portal Exit: (3,1), Goal: (3,3)
    # Path: (0,0) -> R (0,1) -> R (0,2) -> R (0,3) -> D (1,3) [Portal] -> (3,1) -> R (3,2) -> R (3,3) [Goal]
    actions_to_take_p = [
        3, # R: (0,0) -> (0,1)
        3, # R: (0,1) -> (0,2)
        3, # R: (0,2) -> (0,3)
        1, # D: (0,3) -> (1,3) - lands on Portal, teleports to (3,1)
        3, # R: (3,1) -> (3,2)
        3, # R: (3,2) -> (3,3) - Goal!
    ]
    print("Path to take portal:")
    for i, action in enumerate(actions_to_take_p):
        print(f"Step {i+1}: Current obs: {obs_p}, Taking action {action}")
        obs_p, reward, done, _ = env_portal.step(action)
        total_reward_p += reward
        env_portal.render()
        print(f"New Obs: {obs_p}, Reward: {reward:.1f}, Done: {done}")
        if done:
            print("Goal reached via portal!")
            break
    print(f"Total reward (portal path): {total_reward_p:.1f}")

    # Test invalid portal configs
    try:
        TimelineGridEnv(grid_size=(3,3), portals={(0,0):(30,30)})
    except ValueError as e:
        print(f"\nCaught expected error for portal out of bounds: {e}")
    try:
        TimelineGridEnv(grid_size=(3,3), walls=[(0,1)], portals={(0,1):(1,1)})
    except ValueError as e:
        print(f"Caught expected error for portal on wall: {e}")
    try:
        TimelineGridEnv(grid_size=(3,3), goal_pos=(1,1), portals={(1,1):(0,0)})
    except ValueError as e:
        print(f"Caught expected error for portal entry on goal: {e}")
