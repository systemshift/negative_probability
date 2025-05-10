import random
import numpy as np
from qp_rl_project.standard_q_agent import StandardQAgent

class RandomRestartQAgent(StandardQAgent):
    """
    A Q-learning agent with random restarts (jumps to previously visited states).
    
    This agent is similar to a standard Q-learning agent but with an added ability
    to randomly teleport to a previously visited state when it detects it might be
    stuck or performing poorly.
    
    It serves as a baseline to compare against the more sophisticated QP-RL agent,
    which uses negative probabilities to guide its jumps intelligently.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 min_exploration_rate=0.01, q_init_val=0.0,
                 use_softmax=False, softmax_temp=1.0,
                 restart_probability=0.1, restart_threshold=3):
        """
        Initialize a Q-learning agent with random restarts.
        
        Args:
            env: The environment instance
            alpha, gamma, exploration_rate, etc.: Standard Q-learning parameters
            restart_probability (float): Probability of attempting a restart when eligible
            restart_threshold (int): Number of consecutive non-improving steps before restart becomes eligible
        """
        super().__init__(
            env=env,
            alpha=alpha,
            gamma=gamma,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate,
            q_init_val=q_init_val,
            use_softmax=use_softmax,
            softmax_temp=softmax_temp
        )
        
        self.restart_probability = restart_probability
        self.restart_threshold = restart_threshold
        self.non_improving_steps = 0
        self.last_max_q = -float('inf')
    
    def choose_action(self, state, trajectory_history=None):
        """
        Choose an action with potential random restarts.
        
        Args:
            state: Current state
            trajectory_history: History of (state, action, reward) tuples
            
        Returns:
            Action or jump instruction
        """
        # Check for restart condition
        if (trajectory_history and 
            len(trajectory_history) >= self.restart_threshold and 
            self.non_improving_steps >= self.restart_threshold and
            random.random() < self.restart_probability):
            
            # Choose a random past state to jump to
            past_states = [s for s, _, _ in trajectory_history]
            jump_target = random.choice(past_states)
            self.non_improving_steps = 0  # Reset counter after jump
            return ("JUMP", jump_target)
        
        # Regular action selection
        action = super().choose_action(state)
        
        # Update non-improving steps counter
        if state in self.q_table and self.q_table[state]:
            current_max_q = max(self.q_table[state].values())
            if current_max_q <= self.last_max_q:
                self.non_improving_steps += 1
            else:
                self.non_improving_steps = 0
                self.last_max_q = current_max_q
        
        return action
