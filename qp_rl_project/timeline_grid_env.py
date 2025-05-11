import collections

class TimelineGridEnv:
    def __init__(self, grid_size=(5,5), start_pos=(0,0), goal_pos=None, walls=None,
                 one_way_doors=None, portals=None, custom_rewards=None, step_penalty=-0.1,
                 dynamic_obstacles=None, keys=None, doors=None):
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

        self.custom_rewards = custom_rewards if custom_rewards is not None else []
        # Example custom_rewards: [{'position': (r,c), 'reward': float_value}]
        # Basic validation for custom_rewards
        for cr in self.custom_rewards:
            if not all(k in cr for k in ['position', 'reward']):
                raise ValueError(f"Custom reward {cr} is missing 'position' or 'reward' key.")
            pos = cr['position']
            if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                raise ValueError(f"Position {pos} in custom reward {cr} is outside grid boundaries.")
            if not isinstance(cr['reward'], (int, float)):
                raise ValueError(f"Reward value in {cr} must be a number.")
            if pos == self.start_pos:
                raise ValueError(f"Custom reward {cr} cannot be at the start position.")
            # It's okay for a custom reward to be at the goal, it might modify the goal reward.
            # It's okay for a custom reward to be on a wall, though agent might not reach it.

        self.step_penalty = step_penalty
        if not isinstance(self.step_penalty, (int, float)):
            raise ValueError(f"Step penalty {self.step_penalty} must be a number.")

        self.dynamic_obstacles = dynamic_obstacles if dynamic_obstacles is not None else []
        # Example dynamic_obstacles: [{'path': [(r1,c1), (r2,c2), ...], 'period': T}]
        # 'path' defines the sequence of positions the obstacle cycles through.
        # 'period' is how many time steps it stays at each position in its path.
        # Current position of each dynamic obstacle will be stored internally.
        self._current_dynamic_obstacle_positions = []
        for i, do in enumerate(self.dynamic_obstacles):
            if not all(k in do for k in ['path', 'period']):
                raise ValueError(f"Dynamic obstacle {do} is missing 'path' or 'period' key.")
            if not do['path']:
                raise ValueError(f"Dynamic obstacle {do} has an empty path.")
            for pos in do['path']:
                if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                    raise ValueError(f"Position {pos} in dynamic obstacle path {do['path']} is outside grid.")
                if pos in self.walls:
                     raise ValueError(f"Dynamic obstacle path {do['path']} cannot include a wall position {pos}.")
            if not isinstance(do['period'], int) or do['period'] <= 0:
                raise ValueError(f"Dynamic obstacle period {do['period']} must be a positive integer.")
            # Initialize internal state for dynamic obstacles
            self._current_dynamic_obstacle_positions.append(do['path'][0])

        self.keys = keys if keys is not None else [] # List of {'id': key_id, 'position': (r,c)}
        self._active_keys = [] # Internal state for keys present in the env
        for key_spec in self.keys:
            if not all(k in key_spec for k in ['id', 'position']):
                raise ValueError(f"Key {key_spec} is missing 'id' or 'position'.")
            pos = key_spec['position']
            if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                raise ValueError(f"Key position {pos} is outside grid.")
            if pos in self.walls or pos == self.start_pos or pos == self.goal_pos:
                raise ValueError(f"Key {key_spec} cannot be on a wall, start, or goal.")

        self.doors = doors if doors is not None else []
        # List of {'position': (r,c), 'key_id_required': key_id, 'locked': True/False}
        self._door_states = [] # Internal state for door locked status
        for door_spec in self.doors:
            if not all(k in door_spec for k in ['position', 'key_id_required', 'locked']):
                raise ValueError(f"Door {door_spec} is missing 'position', 'key_id_required', or 'locked'.")
            pos = door_spec['position']
            if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                raise ValueError(f"Door position {pos} is outside grid.")
            if pos in self.walls or pos == self.start_pos or pos == self.goal_pos:
                 raise ValueError(f"Door {door_spec} cannot be on a wall, start, or goal.")
            if not isinstance(door_spec['locked'], bool):
                raise ValueError(f"Door 'locked' status for {door_spec} must be boolean.")
        
        self.agent_pos = None # Will be set in reset()
        self.time_step = 0
        self.inventory = set() # Agent's key inventory

    def _update_dynamic_obstacles(self):
        """Updates the positions of dynamic obstacles based on the current time_step."""
        self._current_dynamic_obstacle_positions = []
        for do_spec in self.dynamic_obstacles:
            path = do_spec['path']
            period = do_spec['period']
            # Calculate current index in the path
            # Each position in path is held for 'period' time steps.
            # Total cycle length for one round through path is len(path) * period.
            # Current "phase" in the cycle: self.time_step % (len(path) * period)
            # Index in path: (self.time_step // period) % len(path)
            current_path_idx = (self.time_step // period) % len(path)
            self._current_dynamic_obstacle_positions.append(path[current_path_idx])

    def reset(self):
        """
        Resets the agent to the starting position and time step to 0.
        Resets keys and doors to their initial states.
        Returns the initial observation (agent's position).
        """
        self.agent_pos = self.start_pos
        self.time_step = 0
        self._update_dynamic_obstacles() 
        
        # Reset keys: copy from initial spec to active_keys
        self._active_keys = [dict(k) for k in self.keys] # Make a deep copy for mutable state
        
        # Reset doors: copy from initial spec to _door_states
        self._door_states = [dict(d) for d in self.doors] # Make a deep copy

        self.inventory = set() # Clear agent inventory
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
                break
        
        if not one_way_door_taken:
            next_row = current_row + dr
            next_col = current_col + dc

            # Boundary checks
            next_row = max(0, min(next_row, self.rows - 1))
            next_col = max(0, min(next_col, self.cols - 1))
            
            # Check for wall collision
            if (next_row, next_col) in self.walls:
                next_row, next_col = current_row, current_col 
        
        # Tentative new position
        potential_agent_pos = (next_row, next_col)

        # Check for door interaction
        door_collided = False
        for i, door_state in enumerate(self._door_states):
            if door_state['position'] == potential_agent_pos and door_state['locked']:
                if door_state['key_id_required'] in self.inventory:
                    self._door_states[i]['locked'] = False # Unlock the door
                    # Agent passes through, potential_agent_pos is fine
                    # print(f"DEBUG: Unlocked and passed door at {potential_agent_pos} with key {door_state['key_id_required']}")
                else:
                    # Agent hits a locked door without the key
                    potential_agent_pos = (current_row, current_col) # Stay in place
                    door_collided = True
                    # print(f"DEBUG: Hit locked door at {door_state['position']}. Required key: {door_state['key_id_required']}, Inventory: {self.inventory}")
                break 
        
        self.time_step += 1 # Increment time first
        self._update_dynamic_obstacles() # Update DOs based on new time

        # Check for dynamic obstacle collision
        if potential_agent_pos in self._current_dynamic_obstacle_positions:
            self.agent_pos = (current_row, current_col) # Stay in place before DO
            reward = -2.0 
            done = False
            return self.agent_pos, reward, done, {"collision_type": "dynamic_obstacle"}
        
        # If collided with a locked door (and not overridden by DO collision)
        if door_collided: # This check is after DO check, agent stays at current_row, current_col
            self.agent_pos = (current_row, current_col)
            reward = -1.5 # Penalty for hitting a locked door
            done = False
            return self.agent_pos, reward, done, {"collision_type": "locked_door"}

        # No collision with DO or locked door, update agent position
        self.agent_pos = potential_agent_pos
        
        # Check for key pickup
        key_to_remove_idx = -1
        for i, key_spec in enumerate(self._active_keys):
            if self.agent_pos == key_spec['position']:
                self.inventory.add(key_spec['id'])
                key_to_remove_idx = i
                # print(f"DEBUG: Picked up key {key_spec['id']} at {self.agent_pos}. Inventory: {self.inventory}")
                break 
        if key_to_remove_idx != -1:
            del self._active_keys[key_to_remove_idx] # Key is picked up

        # Check for portal transition
        if self.agent_pos in self.portals:
            self.agent_pos = self.portals[self.agent_pos]

        # Reward logic
        done = False
        reward = self.step_penalty 

        for cr in self.custom_rewards:
            if self.agent_pos == cr['position']:
                reward = cr['reward']
                break 

        if self.agent_pos == self.goal_pos:
            is_goal_custom_rewarded = False
            for cr in self.custom_rewards:
                if self.agent_pos == cr['position'] and self.agent_pos == self.goal_pos:
                    is_goal_custom_rewarded = True
                    break
            if not is_goal_custom_rewarded:
                 reward = 1.0 
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
                    char_to_print = ". " # Default empty space

                    if pos == self.agent_pos:
                        char_to_print = "A "
                    elif pos == self.goal_pos:
                        char_to_print = "G "
                    elif pos in self.walls:
                        char_to_print = "# "
                    elif pos in self._current_dynamic_obstacle_positions:
                        char_to_print = "D "
                    elif any(k['position'] == pos for k in self._active_keys):
                        # Find the key_id to display, perhaps first char or number
                        key_id_str = "K"
                        for k_spec in self._active_keys:
                            if k_spec['position'] == pos:
                                key_id_str = str(k_spec['id'])[0] + " " # Display first char of key ID
                                break
                        char_to_print = key_id_str
                    elif any(d['position'] == pos for d in self._door_states):
                        is_locked = False
                        for d_state in self._door_states:
                            if d_state['position'] == pos:
                                is_locked = d_state['locked']
                                break
                        char_to_print = "L " if is_locked else "U " # Locked/Unlocked Door
                    elif any(cr['position'] == pos for cr in self.custom_rewards):
                        char_to_print = "$ " 
                    elif pos in self.portals: 
                        char_to_print = "P "
                    elif pos in self.portals.values() and pos not in self.portals: 
                        is_also_an_entry = any(val_pos == pos for val_pos in self.portals.keys())
                        if not is_also_an_entry: 
                            char_to_print = "O " 
                    
                    row_str += char_to_print
                print(row_str)
            print(f"Time: {self.time_step} Inv: {sorted(list(self.inventory))}") # Display time and inventory
            print("-" * (self.cols * 2)) 
        else:
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

    print("\n--- Testing TimelineGridEnv with Custom Rewards and Step Penalty (Phase CR) ---")
    custom_reward_list_cr = [
        {'position': (0,2), 'reward': 5.0}, # A highly rewarding spot
        {'position': (1,0), 'reward': -2.0} # A penalty spot
    ]
    env_cr = TimelineGridEnv(grid_size=(3,3), start_pos=(0,0), goal_pos=(2,2),
                             custom_rewards=custom_reward_list_cr, step_penalty=-0.5)
    obs_cr = env_cr.reset()
    env_cr.render()
    total_reward_cr = 0
    
    actions_to_take_cr = [
        3, # R: (0,0) -> (0,1), reward -0.5
        3, # R: (0,1) -> (0,2) (custom reward 5.0)
        1, # D: (0,2) -> (1,2), reward -0.5
        2, # L: (1,2) -> (1,1), reward -0.5
        2, # L: (1,1) -> (1,0) (custom reward -2.0)
        1, # D: (1,0) -> (2,0), reward -0.5
        3, # R: (2,0) -> (2,1), reward -0.5
        3, # R: (2,1) -> (2,2) (Goal, reward 1.0)
    ]
    print("Path to test custom rewards:")
    for i, action in enumerate(actions_to_take_cr):
        print(f"Step {i+1}: Current obs: {obs_cr}, Taking action {action}")
        obs_cr, reward, done, _ = env_cr.step(action)
        total_reward_cr += reward
        env_cr.render()
        print(f"New Obs: {obs_cr}, Reward: {reward:.1f}, Done: {done}")
        if done:
            print("Goal reached with custom rewards!")
            break
    print(f"Total reward (custom rewards path): {total_reward_cr:.1f}")

    # Test invalid custom_rewards configs
    try:
        TimelineGridEnv(grid_size=(3,3), custom_rewards=[{'pos':(0,0), 'reward':1}]) # Wrong key
    except ValueError as e:
        print(f"\nCaught expected error for invalid custom reward key: {e}")
    try:
        TimelineGridEnv(grid_size=(3,3), custom_rewards=[{'position':(10,10), 'reward':1}])
    except ValueError as e:
        print(f"Caught expected error for custom reward out of bounds: {e}")
    try:
        TimelineGridEnv(grid_size=(3,3), step_penalty="high")
    except ValueError as e:
        print(f"Caught expected error for invalid step penalty type: {e}")

    print("\n--- Testing TimelineGridEnv with Dynamic Obstacles (Phase DO) ---")
    dynamic_obstacle_list_do = [
        {'path': [(1,1), (1,2), (1,3), (1,2)], 'period': 1}, # Moves R, R, L, L period 1
        {'path': [(3,2), (2,2), (1,2)], 'period': 2}       # Moves U, U, D period 2
    ]
    env_do = TimelineGridEnv(grid_size=(5,5), start_pos=(0,0), goal_pos=(4,4),
                             walls=[(0,1)], dynamic_obstacles=dynamic_obstacle_list_do)
    obs_do = env_do.reset()
    env_do.render()
    total_reward_do = 0
    
    # Actions to test dynamic obstacles:
    # Agent tries to move into a space that will be occupied by a dynamic obstacle
    # Agent waits for a dynamic obstacle to pass
    actions_to_take_do = [
        1, # D (0,0)->(1,0). Time 1. DO1 at (1,2), DO2 at (2,2)
        3, # R (1,0)->(1,1). Time 2. DO1 at (1,3), DO2 at (2,2). Agent hits DO1 if it moves to (1,2)
           # Let's try to hit it: action 3 (Right)
        3, # R (1,1)->(1,1) (collision with DO1 at (1,2)). Time 3. DO1 at (1,3), DO2 at (1,2). Reward -2.0
        0, # U (1,1)->(0,1) (hits wall). Time 4. DO1 at (1,2), DO2 at (1,2). Reward -1.0
        1, # D (0,1)->(0,1) (wall). Time 5. DO1 at (1,2), DO2 at (3,2).
        3, # R (0,1)->(0,1) (wall). Time 6. DO1 at (1,1), DO2 at (3,2).
        # Let agent move to (1,0)
        1, # D (0,0) from reset. obs_do=(0,0)
        # Reset and try a path that waits
    ]
    env_do.reset() # Reset for a clean test sequence
    obs_do = env_do.agent_pos
    print("Path to test dynamic obstacles (collision and waiting):")
    # Sequence 1: Collision
    print("Sequence 1: Agent attempts to move into DO path")
    # Time 0: Agent (0,0), DO1 (1,1), DO2 (3,2)
    # Action: Down (to (1,0))
    # Time 1: Agent (1,0), DO1 (1,2) (path[1]), DO2 (3,2) (path[0], period 2, time 1//2=0)
    # Action: Right (to (1,1))
    # Time 2: Agent (1,1), DO1 (1,3) (path[2]), DO2 (2,2) (path[1], period 2, time 2//2=1)
    # Action: Right (to (1,2)). DO1 will be at (1,2) at Time 3.
    # Time 3: Agent should collide. DO1 (1,2) (path[0]), DO2 (2,2) (path[1], period 2, time 3//2=1)
    
    test_actions_collision = [1, 3, 3] 
    for i, action in enumerate(test_actions_collision):
        print(f"Step {i+1}: Current obs: {obs_do}, Time: {env_do.time_step}, Taking action {action}")
        print(f"  DOs before step: {env_do._current_dynamic_obstacle_positions}")
        obs_do, reward, done, info = env_do.step(action)
        total_reward_do += reward
        env_do.render()
        print(f"  New Obs: {obs_do}, Reward: {reward:.1f}, Done: {done}, Info: {info}")
        if done: break
    print(f"Total reward after collision test: {total_reward_do:.1f}")

    # Sequence 2: Waiting
    print("\nSequence 2: Agent waits for DO to pass")
    env_do.reset()
    obs_do = env_do.agent_pos
    total_reward_do = 0
    # Time 0: Agent (0,0), DO1 (1,1), DO2 (3,2)
    # Action: Down (to (1,0))
    # Time 1: Agent (1,0), DO1 (1,2), DO2 (3,2)
    # Action: Stay (e.g. try to move into self - needs a 'wait' action or smart move)
    # For now, let's assume agent tries to move Right (to (1,1)) but DO1 is at (1,1) at T=0, T=6 etc.
    # Let's make DO1 path: [(1,1), (2,1)] period 1. So it's at (1,1) at T=0,2,4... and (2,1) at T=1,3,5...
    env_do.dynamic_obstacles = [{'path': [(1,1), (2,1)], 'period': 1}]
    env_do.reset()
    obs_do = env_do.agent_pos # (0,0)
    env_do.render()
    # Time 0: A(0,0), DO(1,1)
    # Action: D to (1,0)
    # Time 1: A(1,0), DO(2,1). Reward -0.1
    print(f"Step 1: Current obs: {obs_do}, Time: {env_do.time_step}, Taking action 1 (Down)")
    obs_do, reward, _, info = env_do.step(1) # Down
    total_reward_do += reward
    env_do.render()
    print(f"  New Obs: {obs_do}, Reward: {reward:.1f}, Info: {info}")
    # Time 1: A(1,0), DO(2,1)
    # Action: R to (1,1). (1,1) is clear.
    # Time 2: A(1,1), DO(1,1). Reward -0.1
    print(f"Step 2: Current obs: {obs_do}, Time: {env_do.time_step}, Taking action 3 (Right)")
    obs_do, reward, _, info = env_do.step(3) # Right
    total_reward_do += reward
    env_do.render()
    print(f"  New Obs: {obs_do}, Reward: {reward:.1f}, Info: {info}")
    # Time 2: A(1,1), DO(1,1)
    # Action: R to (1,2). DO will be at (2,1) at T=3. (1,2) is clear.
    # Time 3: A(1,2), DO(2,1). Reward -0.1
    print(f"Step 3: Current obs: {obs_do}, Time: {env_do.time_step}, Taking action 3 (Right)")
    obs_do, reward, _, info = env_do.step(3) # Right
    total_reward_do += reward
    env_do.render()
    print(f"  New Obs: {obs_do}, Reward: {reward:.1f}, Info: {info}")
    print(f"Total reward after waiting test: {total_reward_do:.1f}")

    # Test invalid dynamic_obstacles configs
    try:
        TimelineGridEnv(grid_size=(3,3), dynamic_obstacles=[{'path':[(0,0)], 'period':0}]) 
    except ValueError as e:
        print(f"\nCaught expected error for invalid DO period: {e}")
    try:
        TimelineGridEnv(grid_size=(3,3), dynamic_obstacles=[{'path':[(0,0),(10,10)], 'period':1}])
    except ValueError as e:
        print(f"Caught expected error for DO path out of bounds: {e}")

    print("\n--- Testing TimelineGridEnv with Keys and Doors (Phase KD) ---")
    keys_kd = [
        {'id': 'red', 'position': (0,1)},
        {'id': 'blue', 'position': (2,0)}
    ]
    doors_kd = [
        {'position': (1,2), 'key_id_required': 'red', 'locked': True},
        {'position': (2,2), 'key_id_required': 'blue', 'locked': True}
    ]
    env_kd = TimelineGridEnv(grid_size=(3,3), start_pos=(0,0), goal_pos=(2,2),
                             walls=[(1,1)], keys=keys_kd, doors=doors_kd)
    obs_kd = env_kd.reset()
    env_kd.render()
    total_reward_kd = 0

    # Path: Get red key, open red door, get blue key, open blue door (goal)
    # (0,0) -> R (0,1) [Pick Red Key] -> D (1,1) [Wall] -> R (0,1) -> D (1,1) [Wall]
    # (0,0) -> R (0,1) [Pick Red Key 'red']
    # (0,1) -> D (1,1) [Wall, stay (0,1)]
    # (0,1) -> D to (1,1) is wall.
    # (0,1) -> R to (0,2)
    # (0,2) -> D to (1,2) [Open Red Door]
    # (1,2) -> L to (1,1) [Wall, stay (1,2)]
    # (1,2) -> D to (2,2) [Goal, but Blue Door is there]
    # Need to get Blue key from (2,0)
    # Path: (0,0) -> R (0,1)[Key R] -> D (to (1,1) but wall) -> (0,1)
    #       (0,1) -> D (to (1,1) wall)
    #       (0,1) -> L (to (0,0))
    #       (0,0) -> D (to (1,0))
    #       (1,0) -> D (to (2,0)) [Key B]
    #       (2,0) -> R (to (2,1))
    #       (2,1) -> R (to (2,2)) [Open Blue Door, Goal]
    # This path does not use red door. Let's simplify.
    # Goal (0,2). Red Door at (0,2). Red Key at (0,1). Start (0,0)
    env_kd.goal_pos = (0,2)
    env_kd.keys = [{'id': 'R', 'position': (0,1)}]
    env_kd.doors = [{'position': (0,2), 'key_id_required': 'R', 'locked': True}]
    env_kd.walls = [] # Clear walls for this simple test
    obs_kd = env_kd.reset()
    env_kd.render()

    actions_to_take_kd = [
        3, # R (0,0)->(0,1) [Pick Key R]
        3, # R (0,1)->(0,2) [Open Door R, Goal]
    ]
    print("Path to test keys and doors:")
    for i, action in enumerate(actions_to_take_kd):
        print(f"Step {i+1}: Obs={obs_kd}, Time={env_kd.time_step}, Inv={env_kd.inventory}, Taking action {action}")
        obs_kd, reward, done, info = env_kd.step(action)
        total_reward_kd += reward
        env_kd.render()
        print(f"  New Obs: {obs_kd}, Reward: {reward:.1f}, Done: {done}, Info: {info}")
        if done: break
    print(f"Total reward (keys/doors path): {total_reward_kd:.1f}")

    # Test hitting a locked door
    env_kd.reset() # Resets inventory and door locks
    obs_kd = env_kd.agent_pos
    print("\nTest hitting locked door:")
    print(f"Step 1: Obs={obs_kd}, Time={env_kd.time_step}, Inv={env_kd.inventory}, Taking action 3 (Right to key)")
    obs_kd, reward, _, _ = env_kd.step(3) # Get key
    env_kd.render()
    # Now try to open door again, but reset inventory
    env_kd.inventory = set() # Manually remove key for test
    print(f"Step 2: Obs={obs_kd}, Time={env_kd.time_step}, Inv={env_kd.inventory} (Key Removed!), Taking action 3 (Right to door)")
    obs_kd, reward, _, info = env_kd.step(3) # Try to open door
    env_kd.render()
    print(f"  New Obs: {obs_kd}, Reward: {reward:.1f}, Info: {info}") # Should be at (0,1), reward -1.5


    print("\n--- Testing TimelineGridEnv with One-Way Doors (Phase 3) ---")
    one_way_door_list_p3 = [
        {'from': (1,2), 'to': (2,2), 'action': 1} 
    ]
    env_owd = TimelineGridEnv(grid_size=(4,4), start_pos=(0,1), goal_pos=(2,3), 
                              walls=[(1,1)], one_way_doors=one_way_door_list_p3, step_penalty=-0.1) # Added default step_penalty for consistency
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
