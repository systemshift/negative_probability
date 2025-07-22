import numpy as np
from typing import List, Tuple, Dict, Optional
import collections

class TemporalNavigationEnv:
    """
    A physics-based grid world environment with reversible dynamics.
    
    State representation: (x, y, vx, vy, t)
    - (x, y): position on grid
    - (vx, vy): velocity components
    - t: time step
    
    Actions apply forces (accelerations) to change velocity.
    Position updates based on velocity, making dynamics reversible.
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (10, 10),
                 max_velocity: float = 2.0,
                 friction: float = 0.1,
                 goal_pos: Optional[Tuple[int, int]] = None,
                 obstacles: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize the temporal navigation environment.
        
        Args:
            grid_size: (rows, cols) of the grid
            max_velocity: Maximum velocity magnitude in any direction
            friction: Friction coefficient (0-1), reduces velocity each step
            goal_pos: Goal position (x, y)
            obstacles: List of obstacle positions
        """
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.max_velocity = max_velocity
        self.friction = friction
        
        # Goal position
        if goal_pos is None:
            self.goal_pos = (self.rows - 1, self.cols - 1)
        else:
            self.goal_pos = goal_pos
            
        # Obstacles
        self.obstacles = obstacles if obstacles is not None else []
        
        # Action space: 9 actions (including no action)
        # Each action applies acceleration in a direction
        self.actions = {
            0: (0, 0),    # No acceleration
            1: (-1, 0),   # Up
            2: (1, 0),    # Down
            3: (0, -1),   # Left
            4: (0, 1),    # Right
            5: (-1, -1),  # Up-Left
            6: (-1, 1),   # Up-Right
            7: (1, -1),   # Down-Left
            8: (1, 1),    # Down-Right
        }
        self.action_space_size = len(self.actions)
        
        # Acceleration magnitude
        self.acceleration = 0.5
        
        # Current state
        self.state = None
        self.time_step = 0
        
    def reset(self, start_pos: Optional[Tuple[int, int]] = None) -> Tuple[int, int, float, float, int]:
        """Reset environment to initial state."""
        if start_pos is None:
            start_pos = (0, 0)
        
        # Start with zero velocity
        self.state = (*start_pos, 0.0, 0.0, 0)
        self.time_step = 0
        
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int, float, float, int], float, bool, Dict]:
        """
        Execute one time step with given action.
        
        Returns:
            state: New state (x, y, vx, vy, t)
            reward: Reward for this transition
            done: Whether episode is complete
            info: Additional information
        """
        if self.state is None:
            raise ValueError("Environment must be reset before stepping")
            
        x, y, vx, vy, t = self.state
        
        # Apply acceleration from action
        ax, ay = self.actions[action]
        ax *= self.acceleration
        ay *= self.acceleration
        
        # Update velocity with acceleration and friction
        new_vx = vx + ax
        new_vy = vy + ay
        
        # Apply friction
        new_vx *= (1 - self.friction)
        new_vy *= (1 - self.friction)
        
        # Clamp velocity to max
        velocity_magnitude = np.sqrt(new_vx**2 + new_vy**2)
        if velocity_magnitude > self.max_velocity:
            scale = self.max_velocity / velocity_magnitude
            new_vx *= scale
            new_vy *= scale
        
        # Update position based on velocity
        new_x = x + new_vx
        new_y = y + new_vy
        
        # Handle boundaries (bounce with velocity reversal)
        if new_x < 0:
            new_x = 0
            new_vx = -new_vx * 0.5  # Lose some energy on bounce
        elif new_x >= self.rows:
            new_x = self.rows - 1
            new_vx = -new_vx * 0.5
            
        if new_y < 0:
            new_y = 0
            new_vy = -new_vy * 0.5
        elif new_y >= self.cols:
            new_y = self.cols - 1
            new_vy = -new_vy * 0.5
        
        # Round position to integer grid coordinates
        new_x = int(round(new_x))
        new_y = int(round(new_y))
        
        # Check obstacle collision
        if (new_x, new_y) in self.obstacles:
            # Bounce back
            new_x, new_y = x, y
            new_vx = -vx * 0.5
            new_vy = -vy * 0.5
            
        # Update time
        new_t = t + 1
        
        # Update state
        self.state = (new_x, new_y, new_vx, new_vy, new_t)
        self.time_step = new_t
        
        # Calculate reward
        reward = self._calculate_reward(self.state, action)
        
        # Check if done
        done = (new_x, new_y) == self.goal_pos
        
        info = {
            'velocity_magnitude': np.sqrt(new_vx**2 + new_vy**2),
            'action_taken': action
        }
        
        return self.state, reward, done, info
    
    def _calculate_reward(self, state: Tuple[int, int, float, float, int], action: int) -> float:
        """Calculate reward for reaching a state."""
        x, y, vx, vy, t = state
        
        # Goal reward
        if (x, y) == self.goal_pos:
            return 10.0
        
        # Distance-based shaping reward
        dist_to_goal = np.sqrt((x - self.goal_pos[0])**2 + (y - self.goal_pos[1])**2)
        
        # Small penalty for time and small reward for getting closer
        reward = -0.1 - dist_to_goal * 0.01
        
        # Penalty for high velocity (encourage control)
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        if velocity_magnitude > self.max_velocity * 0.8:
            reward -= 0.1
            
        return reward
    
    def get_predecessor_states(self, state: Tuple[int, int, float, float, int], 
                              num_samples: int = 10) -> List[Tuple[Tuple[int, int, float, float, int], int, float]]:
        """
        Compute plausible predecessor states that could have led to the given state.
        
        Returns list of (predecessor_state, action, probability) tuples.
        """
        x, y, vx, vy, t = state
        predecessors = []
        
        if t <= 0:
            return predecessors
        
        # For each possible action, compute what state would lead to current state
        for action, (ax, ay) in self.actions.items():
            # Reverse the dynamics equations
            # Current: new_vx = (old_vx + ax) * (1 - friction)
            # Reverse: old_vx = new_vx / (1 - friction) - ax
            
            ax_scaled = ax * self.acceleration
            ay_scaled = ay * self.acceleration
            
            # Compute previous velocity (before friction and acceleration)
            prev_vx = vx / (1 - self.friction) - ax_scaled
            prev_vy = vy / (1 - self.friction) - ay_scaled
            
            # Compute previous position
            prev_x = x - prev_vx
            prev_y = y - prev_vy
            
            # Check if previous position is valid
            if (0 <= prev_x < self.rows and 0 <= prev_y < self.cols and
                (int(round(prev_x)), int(round(prev_y))) not in self.obstacles):
                
                prev_state = (int(round(prev_x)), int(round(prev_y)), prev_vx, prev_vy, t - 1)
                
                # Compute probability based on how "natural" this transition is
                # Higher probability for smaller accelerations (more likely)
                acceleration_magnitude = np.sqrt(ax_scaled**2 + ay_scaled**2)
                probability = np.exp(-acceleration_magnitude * 2)  # Exponential decay
                
                predecessors.append((prev_state, action, probability))
        
        # Add some noise/variations for more diverse predecessors
        if len(predecessors) < num_samples:
            base_predecessors = predecessors.copy()
            for _ in range(num_samples - len(predecessors)):
                if base_predecessors:
                    # Take a random predecessor and add small noise
                    base_pred, base_action, base_prob = base_predecessors[np.random.randint(len(base_predecessors))]
                    bx, by, bvx, bvy, bt = base_pred
                    
                    # Add small velocity noise
                    noise_vx = np.random.normal(0, 0.1)
                    noise_vy = np.random.normal(0, 0.1)
                    
                    noisy_state = (bx, by, bvx + noise_vx, bvy + noise_vy, bt)
                    predecessors.append((noisy_state, base_action, base_prob * 0.8))
        
        # Normalize probabilities
        total_prob = sum(p for _, _, p in predecessors)
        if total_prob > 0:
            predecessors = [(s, a, p/total_prob) for s, a, p in predecessors]
            
        return predecessors
    
    def get_successor_states(self, state: Tuple[int, int, float, float, int]) -> List[Tuple[Tuple[int, int, float, float, int], int, float]]:
        """
        Compute possible successor states from the given state.
        
        Returns list of (successor_state, action, probability) tuples.
        """
        # Temporarily set state
        old_state = self.state
        old_time = self.time_step
        self.state = state
        self.time_step = state[4]
        
        successors = []
        
        for action in range(self.action_space_size):
            # Simulate taking this action
            self.state = state  # Reset to input state
            next_state, reward, done, info = self.step(action)
            
            # Probability based on action "naturalness"
            ax, ay = self.actions[action]
            acceleration_magnitude = np.sqrt((ax * self.acceleration)**2 + (ay * self.acceleration)**2)
            probability = np.exp(-acceleration_magnitude * 2)
            
            successors.append((next_state, action, probability))
        
        # Restore original state
        self.state = old_state
        self.time_step = old_time
        
        # Normalize probabilities
        total_prob = sum(p for _, _, p in successors)
        if total_prob > 0:
            successors = [(s, a, p/total_prob) for s, a, p in successors]
            
        return successors
    
    def compute_transition_probability(self, s1: Tuple[int, int, float, float, int], 
                                     s2: Tuple[int, int, float, float, int], 
                                     forward: bool = True) -> float:
        """
        Compute the probability of transitioning from s1 to s2.
        
        Args:
            s1: Source state
            s2: Target state  
            forward: If True, compute P(s2|s1). If False, compute P(s1|s2).
        """
        if forward:
            successors = self.get_successor_states(s1)
            for succ_state, action, prob in successors:
                if self._states_equal(succ_state, s2):
                    return prob
        else:
            predecessors = self.get_predecessor_states(s2)
            for pred_state, action, prob in predecessors:
                if self._states_equal(pred_state, s1):
                    return prob
                    
        return 0.0
    
    def _states_equal(self, s1: Tuple[int, int, float, float, int], 
                      s2: Tuple[int, int, float, float, int], 
                      position_only: bool = False) -> bool:
        """Check if two states are equal."""
        if position_only:
            return s1[0] == s2[0] and s1[1] == s2[1]
        else:
            # Allow small differences in velocity due to floating point
            return (s1[0] == s2[0] and s1[1] == s2[1] and 
                   abs(s1[2] - s2[2]) < 0.01 and abs(s1[3] - s2[3]) < 0.01 and
                   s1[4] == s2[4])
    
    def render(self, mode='human', current_state=None):
        """Render the environment."""
        if mode == 'human':
            state = current_state if current_state is not None else self.state
            if state is None:
                print("Environment not initialized")
                return
                
            x, y, vx, vy, t = state
            
            print(f"\nTime: {t}")
            print(f"Velocity: ({vx:.2f}, {vy:.2f})")
            
            for r in range(self.rows):
                row_str = ""
                for c in range(self.cols):
                    if (r, c) == (x, y):
                        row_str += "A "
                    elif (r, c) == self.goal_pos:
                        row_str += "G "
                    elif (r, c) in self.obstacles:
                        row_str += "# "
                    else:
                        row_str += ". "
                print(row_str)
            print("-" * (self.cols * 2))


if __name__ == "__main__":
    # Test the environment
    print("Testing TemporalNavigationEnv...")
    
    env = TemporalNavigationEnv(
        grid_size=(5, 5),
        max_velocity=2.0,
        friction=0.1,
        goal_pos=(4, 4),
        obstacles=[(2, 2), (2, 3), (3, 2)]
    )
    
    # Test forward dynamics
    print("\n=== Testing Forward Dynamics ===")
    state = env.reset((0, 0))
    print(f"Initial state: {state}")
    env.render()
    
    # Take some actions
    actions_to_test = [8, 8, 4, 2]  # Down-Right, Down-Right, Right, Down
    for i, action in enumerate(actions_to_test):
        print(f"\nStep {i+1}: Taking action {action} ({list(env.actions.values())[action]})")
        state, reward, done, info = env.step(action)
        print(f"New state: {state}")
        print(f"Reward: {reward:.3f}, Done: {done}")
        env.render()
        
        if done:
            print("Goal reached!")
            break
    
    # Test predecessor computation
    print("\n=== Testing Predecessor States ===")
    current_state = state
    predecessors = env.get_predecessor_states(current_state, num_samples=5)
    print(f"\nCurrent state: {current_state}")
    print(f"Found {len(predecessors)} predecessor states:")
    for pred_state, action, prob in predecessors[:5]:
        print(f"  State: {pred_state}, Action: {action}, Prob: {prob:.3f}")
    
    # Test successor computation
    print("\n=== Testing Successor States ===")
    successors = env.get_successor_states(current_state)
    print(f"\nCurrent state: {current_state}")
    print(f"Found {len(successors)} successor states:")
    for succ_state, action, prob in successors[:5]:
        print(f"  State: {succ_state}, Action: {action}, Prob: {prob:.3f}")
    
    # Test transition probability
    print("\n=== Testing Transition Probabilities ===")
    if predecessors:
        pred_state, _, _ = predecessors[0]
        forward_prob = env.compute_transition_probability(pred_state, current_state, forward=True)
        backward_prob = env.compute_transition_probability(pred_state, current_state, forward=False)
        print(f"P(current|predecessor) = {forward_prob:.3f}")
        print(f"P(predecessor|current) = {backward_prob:.3f}")
