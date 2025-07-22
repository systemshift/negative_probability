import numpy as np
import collections
from typing import Dict, List, Tuple, Optional, Union

class TemporalQPAgent:
    """
    Temporal Quasi-Probability Agent that maintains a probability distribution
    over state-time space, with positive probabilities for future states
    and negative probabilities for past states.
    
    Key concepts:
    - Current state has probability 1.0
    - Future states have positive probabilities (sum < 1)
    - Past states have negative probabilities (sum > -1)
    - Total quasi-probability sums to 1.0
    """
    
    def __init__(self, 
                 env,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 0.3,
                 exploration_decay: float = 0.995,
                 min_exploration: float = 0.01,
                 temporal_horizon: int = 5,
                 negative_prob_weight: float = 0.3):
        """
        Initialize the temporal QP agent.
        
        Args:
            env: The TemporalNavigationEnv instance
            learning_rate: Learning rate for value updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate
            exploration_decay: Decay rate for exploration
            min_exploration: Minimum exploration rate
            temporal_horizon: How many steps forward/backward to consider
            negative_prob_weight: Weight given to past states (0-1)
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.temporal_horizon = temporal_horizon
        self.negative_prob_weight = negative_prob_weight
        
        # Value function: maps states to expected values
        self.value_function = collections.defaultdict(float)
        
        # Quasi-probability distribution over state-time space
        self.quasi_prob_distribution = {}
        
        # Current state
        self.current_state = None
        
        # Episode history for learning
        self.episode_history = []
        
        # Statistics
        self.backward_jumps = 0
        self.forward_steps = 0
        
    def reset(self, initial_state):
        """Reset agent for new episode."""
        self.current_state = initial_state
        self.episode_history = [(initial_state, None, 0)]
        self.backward_jumps = 0
        self.forward_steps = 0
        
        # Initialize quasi-probability distribution
        self._update_quasi_probability_distribution()
        
        return self.current_state
    
    def _update_quasi_probability_distribution(self):
        """
        Update the quasi-probability distribution centered on current state.
        
        Distribution structure:
        - Current state: P = 1.0
        - Future states: P > 0 (expanding cone)
        - Past states: P < 0 (expanding cone)
        """
        self.quasi_prob_distribution = {}
        
        if self.current_state is None:
            return
            
        # Current state has probability 1
        self.quasi_prob_distribution[self.current_state] = 1.0
        
        # Compute future states (positive probabilities)
        future_states = self._compute_future_cone(self.current_state, self.temporal_horizon)
        
        # Compute past states (negative probabilities)
        past_states = self._compute_past_cone(self.current_state, self.temporal_horizon)
        
        # Normalize to ensure sum = 1
        self._normalize_quasi_probabilities(future_states, past_states)
        
    def _compute_future_cone(self, state, horizon):
        """Compute future states and their positive probabilities."""
        future_states = {}
        
        # BFS to explore future states
        queue = [(state, 0, 1.0)]
        visited = {state}
        
        while queue:
            current, depth, prob = queue.pop(0)
            
            if depth >= horizon:
                continue
                
            # Get successor states
            successors = self.env.get_successor_states(current)
            
            for next_state, action, trans_prob in successors:
                if next_state not in visited:
                    visited.add(next_state)
                    
                    # Probability decays with depth
                    future_prob = prob * trans_prob * (0.8 ** depth)
                    
                    if next_state in future_states:
                        future_states[next_state] += future_prob
                    else:
                        future_states[next_state] = future_prob
                        
                    queue.append((next_state, depth + 1, future_prob))
                    
        return future_states
    
    def _compute_past_cone(self, state, horizon):
        """Compute past states and their negative probabilities."""
        past_states = {}
        
        # BFS to explore past states
        queue = [(state, 0, -1.0)]
        visited = {state}
        
        while queue:
            current, depth, prob = queue.pop(0)
            
            if depth >= horizon:
                continue
                
            # Get predecessor states
            predecessors = self.env.get_predecessor_states(current)
            
            for prev_state, action, trans_prob in predecessors:
                if prev_state not in visited:
                    visited.add(prev_state)
                    
                    # Negative probability decays with depth
                    past_prob = prob * trans_prob * (0.8 ** depth)
                    
                    if prev_state in past_states:
                        past_states[prev_state] += past_prob
                    else:
                        past_states[prev_state] = past_prob
                        
                    queue.append((prev_state, depth + 1, past_prob))
                    
        return past_states
    
    def _normalize_quasi_probabilities(self, future_states, past_states):
        """Normalize the quasi-probability distribution to sum to 1."""
        # Add future states (positive probabilities)
        total_future = sum(future_states.values())
        if total_future > 0:
            # Scale future probabilities to use (1 - negative_prob_weight) of positive space
            scale = (1 - self.negative_prob_weight) / total_future
            for state, prob in future_states.items():
                self.quasi_prob_distribution[state] = prob * scale
        
        # Add past states (negative probabilities)
        total_past = sum(abs(p) for p in past_states.values())
        if total_past > 0:
            # Scale past probabilities to use negative_prob_weight of negative space
            scale = self.negative_prob_weight / total_past
            for state, prob in past_states.items():
                self.quasi_prob_distribution[state] = -abs(prob) * scale
    
    def choose_action(self) -> Union[int, Tuple[str, Tuple]]:
        """
        Choose action based on quasi-probability distribution.
        
        Returns:
            Either an action index (for forward movement)
            or ("JUMP", target_state) for backward jump
        """
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.env.action_space_size)
        
        # Exploitation: sample from quasi-probability distribution
        states = list(self.quasi_prob_distribution.keys())
        probs = list(self.quasi_prob_distribution.values())
        
        # Convert to sampling probabilities (handle negative values)
        # Use absolute values for sampling, but track sign
        abs_probs = [abs(p) for p in probs]
        total_abs = sum(abs_probs)
        
        if total_abs == 0:
            return np.random.randint(self.env.action_space_size)
            
        sample_probs = [p / total_abs for p in abs_probs]
        
        # Sample a state
        sampled_idx = np.random.choice(len(states), p=sample_probs)
        sampled_state = states[sampled_idx]
        original_prob = probs[sampled_idx]
        
        # If current state sampled, choose best action
        if sampled_state == self.current_state:
            return self._get_best_action(self.current_state)
        
        # If positive probability (future state), find action to move toward it
        if original_prob > 0:
            # Find best action to move toward sampled future state
            best_action = self._get_action_toward_state(self.current_state, sampled_state)
            self.forward_steps += 1
            return best_action
        
        # If negative probability (past state), jump backward
        else:
            self.backward_jumps += 1
            return ("JUMP", sampled_state)
    
    def _get_best_action(self, state):
        """Get best action based on value function."""
        successors = self.env.get_successor_states(state)
        
        if not successors:
            return np.random.randint(self.env.action_space_size)
        
        best_value = float('-inf')
        best_action = 0
        
        for next_state, action, prob in successors:
            value = self.value_function[next_state]
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action
    
    def _get_action_toward_state(self, current_state, target_state):
        """Find action that moves toward target state."""
        successors = self.env.get_successor_states(current_state)
        
        best_action = 0
        min_distance = float('inf')
        
        for next_state, action, prob in successors:
            # Simple distance metric (could be improved)
            dist = self._state_distance(next_state, target_state)
            if dist < min_distance:
                min_distance = dist
                best_action = action
                
        return best_action
    
    def _state_distance(self, s1, s2):
        """Compute distance between two states."""
        x1, y1, vx1, vy1, t1 = s1
        x2, y2, vx2, vy2, t2 = s2
        
        # Weighted distance considering position and velocity
        pos_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        vel_dist = np.sqrt((vx1 - vx2)**2 + (vy1 - vy2)**2)
        
        return pos_dist + 0.3 * vel_dist
    
    def update(self, state, action, reward, next_state, done):
        """Update value function and quasi-probability distribution."""
        # Update value function (standard TD learning)
        current_value = self.value_function[state]
        next_value = 0 if done else self.value_function[next_state]
        
        td_error = reward + self.discount_factor * next_value - current_value
        self.value_function[state] += self.learning_rate * td_error
        
        # Update current state
        self.current_state = next_state
        
        # Add to history
        self.episode_history.append((next_state, action, reward))
        
        # Update quasi-probability distribution
        self._update_quasi_probability_distribution()
        
        # Decay exploration
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
    
    def get_statistics(self):
        """Get agent statistics."""
        return {
            'backward_jumps': self.backward_jumps,
            'forward_steps': self.forward_steps,
            'exploration_rate': self.exploration_rate,
            'value_function_size': len(self.value_function),
            'current_quasi_prob_size': len(self.quasi_prob_distribution)
        }


if __name__ == "__main__":
    # Test the agent
    from temporal_nav_env import TemporalNavigationEnv
    
    print("Testing TemporalQPAgent...")
    
    # Create environment
    env = TemporalNavigationEnv(
        grid_size=(5, 5),
        max_velocity=2.0,
        friction=0.1,
        goal_pos=(4, 4),
        obstacles=[(2, 2)]
    )
    
    # Create agent
    agent = TemporalQPAgent(
        env,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.5,
        temporal_horizon=3,
        negative_prob_weight=0.3
    )
    
    # Run a short episode
    state = env.reset()
    agent.reset(state)
    
    print(f"Initial state: {state}")
    env.render()
    
    total_reward = 0
    max_steps = 50
    
    for step in range(max_steps):
        # Choose action
        action_or_jump = agent.choose_action()
        
        if isinstance(action_or_jump, tuple) and action_or_jump[0] == "JUMP":
            # Handle backward jump
            _, target_state = action_or_jump
            print(f"\nStep {step}: BACKWARD JUMP to {target_state}")
            
            # Set environment to target state
            env.state = target_state
            env.time_step = target_state[4]
            state = target_state
            agent.current_state = target_state
            
            # Update quasi-probability distribution after jump
            agent._update_quasi_probability_distribution()
            
            env.render()
            continue
        
        # Regular forward action
        action = action_or_jump
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {step}: Action {action}, Reward {reward:.3f}")
        
        # Update agent
        agent.update(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        env.render()
        
        if done:
            print(f"\nGoal reached! Total reward: {total_reward:.3f}")
            break
    
    # Print statistics
    stats = agent.get_statistics()
    print(f"\nAgent Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show quasi-probability distribution
    print(f"\nFinal Quasi-Probability Distribution (top 10):")
    sorted_probs = sorted(agent.quasi_prob_distribution.items(), 
                         key=lambda x: abs(x[1]), reverse=True)[:10]
    for state, prob in sorted_probs:
        sign = "+" if prob >= 0 else ""
        print(f"  State {state}: {sign}{prob:.3f}")
