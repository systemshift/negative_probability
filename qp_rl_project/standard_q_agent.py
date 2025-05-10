import random
import numpy as np
from qp_rl_project.qp_agent import QPAgent

class StandardQAgent(QPAgent):
    """
    A standard Q-learning agent without backward jumps or negative probabilities.
    This agent serves as a baseline to compare against the QP-RL agent.
    
    The key differences from QPAgent:
    1. No backward updates (doesn't use negative Q-values for past plausibility)
    2. No backward jumps (always takes actions forward)
    3. Uses epsilon-greedy instead of softmax for action selection
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 min_exploration_rate=0.01, q_init_val=0.0,
                 use_softmax=False, softmax_temp=1.0):
        """
        Initialize a standard Q-learning agent.
        
        Args:
            env: The environment instance
            alpha (float): Learning rate
            gamma (float): Discount factor for future rewards
            exploration_rate (float): Initial epsilon for epsilon-greedy exploration
            exploration_decay (float): Multiplicative factor to decay exploration_rate
            min_exploration_rate (float): Minimum exploration rate
            q_init_val (float): Initial value for Q-table entries
            use_softmax (bool): Whether to use softmax for action selection (False=epsilon-greedy)
            softmax_temp (float): Temperature for softmax if use_softmax is True
        """
        # Call parent constructor but don't use backward_epsilon or min_trajectory_for_jump
        super().__init__(
            env=env,
            alpha=alpha,
            gamma=gamma,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate,
            q_init_val=q_init_val,
            backward_epsilon=0,  # Not used
            softmax_temp=softmax_temp,
            min_trajectory_for_jump=0  # Not used
        )
        
        self.use_softmax = use_softmax
    
    def choose_action(self, state, trajectory_history=None):
        """
        Choose an action using epsilon-greedy or softmax, but never jump backward.
        
        Args:
            state: Current state
            trajectory_history: Ignored in this implementation
            
        Returns:
            int: The action to take
        """
        if random.random() < self.exploration_rate:
            # Explore: choose a random action
            return random.randrange(self.action_space_size)
        else:
            # Exploit: choose best action (or softmax if enabled)
            state_q_values_dict = self.q_table[state]
            if not state_q_values_dict:
                return random.randrange(self.action_space_size)
                
            actions = list(state_q_values_dict.keys())
            q_values = np.array([state_q_values_dict[a] for a in actions])
            
            if not self.use_softmax:
                # Epsilon-greedy: choose action with highest Q-value
                max_q = np.max(q_values)
                best_actions = [actions[i] for i, q_val in enumerate(q_values) if q_val == max_q]
                return random.choice(best_actions)
            else:
                # Softmax selection
                if self.softmax_temp <= 0:  # Greedy selection if temp is invalid
                    max_q = np.max(q_values)
                    best_actions = [actions[i] for i, q_val in enumerate(q_values) if q_val == max_q]
                    return random.choice(best_actions)
                
                # Use softmax with temperature
                q_values_stable = q_values - np.max(q_values)
                exp_q_values = np.exp(q_values_stable / self.softmax_temp)
                probs = exp_q_values / np.sum(exp_q_values)
                
                if np.isclose(np.sum(probs), 0.0) or not np.all(np.isfinite(probs)):
                    return random.randrange(self.action_space_size)
                    
                return random.choices(actions, weights=probs, k=1)[0]
    
    def learn(self, s, a, r, s2, done_s2, trajectory=None):
        """
        Update Q-values using standard Q-learning (no backward updates).
        
        Args:
            s: Current state
            a: Action taken
            r: Reward received
            s2: Next state
            done_s2: Whether s2 is terminal
            trajectory: Ignored in this implementation
        """
        # Only perform forward update, no backward update
        current_q = self.q_table[s][a]
        
        max_next_q = 0.0
        if not done_s2 and self.q_table[s2]:
            max_next_q = max(self.q_table[s2].values())
            
        target_q = r + self.gamma * max_next_q
        self.q_table[s][a] = current_q + self.alpha * (target_q - current_q)
