import collections
import random
import numpy as np

class QPAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 min_exploration_rate=0.01, q_init_val=0.0, 
                 backward_epsilon=0.01):
        """
        Quasi-Probability Reinforcement Learning Agent.

        Args:
            env: The environment instance (e.g., TimelineGridEnv).
            alpha (float): Learning rate.
            gamma (float): Discount factor for future rewards.
            exploration_rate (float): Initial epsilon for epsilon-greedy exploration.
            exploration_decay (float): Multiplicative factor to decay exploration_rate.
            min_exploration_rate (float): The minimum value for exploration_rate.
            q_init_val (float): Initial value for Q-table entries.
            backward_epsilon (float): The 'eps' in the backward update rule.
                                     abs(q[sp][ap]) will tend towards this value for past states.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma # Note: gamma is not used in the readme's simple forward update
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_init_val = q_init_val
        self.backward_epsilon = backward_epsilon

        # Q-table: stores q(s,a) which are real numbers.
        # Positive values might indicate "good future action".
        # Negative values might indicate "plausible past action".
        self.q_table = collections.defaultdict(
            lambda: collections.defaultdict(lambda: self.q_init_val)
        )
        
        # Ensure action_space_size is available from env
        if not hasattr(env, 'action_space_size'):
            raise AttributeError("The provided environment must have an 'action_space_size' attribute.")
        self.action_space_size = env.action_space_size

    def choose_action(self, state, trajectory_history=None):
        """
        Chooses an action based on the current state using an epsilon-greedy strategy.
        For now, this only implements standard forward action selection.
        The 'backward jump' based on negative Q-values is not yet implemented.

        Args:
            state: The current state of the agent.
            trajectory_history: Not used in this basic version, but planned for backward jumps.

        Returns:
            int: The action to take.
        """
        if random.random() < self.exploration_rate:
            # Explore: choose a random action
            return random.randrange(self.action_space_size)
        else:
            # Exploit: choose the best action from Q-table
            state_q_values = self.q_table[state]
            if not state_q_values: # If no q-values for this state yet
                return random.randrange(self.action_space_size)
            
            # Find the action(s) with the maximum Q-value
            max_q = -float('inf')
            best_actions = []
            for action, q_value in state_q_values.items():
                if q_value > max_q:
                    max_q = q_value
                    best_actions = [action]
                elif q_value == max_q:
                    best_actions.append(action)
            
            if not best_actions: # Should not happen if state_q_values is not empty
                 return random.randrange(self.action_space_size)
            return random.choice(best_actions) # Break ties randomly

    def learn(self, s, a, r, s2, trajectory):
        """
        Updates the Q-table based on the transition and trajectory.
        Implements both forward (future) and backward (past) updates.

        Args:
            s: Current state.
            a: Action taken.
            r: Reward received.
            s2: Next state.
            trajectory (list): History of (state, action, reward) tuples for the current episode.
                               Assumes `(s, a, r)` of the current transition has ALREADY been added.
        """
        # Forward Update (Future) - based on readme: q[s][a] += alpha * (r - q[s][a])
        # This is simpler than a full Q-learning update involving max_q(s2,a').
        # It treats r as the target, somewhat like a contextual bandit.
        current_q_sa = self.q_table[s][a]
        self.q_table[s][a] = current_q_sa + self.alpha * (r - current_q_sa)

        # Backward Update (Past) - q[sp][ap] -= alpha * (abs(q[sp][ap]) - eps)
        # This update applies to the state-action pair (sp, ap) that led to state s.
        if len(trajectory) > 1: # Need at least two entries to get (sp, ap)
            # trajectory[-1] is (s, a, r)
            # trajectory[-2] is (sp, ap, rp)
            sp, ap, _ = trajectory[-2] 
            
            q_val_past = self.q_table[sp][ap]
            self.q_table[sp][ap] = q_val_past - self.alpha * (abs(q_val_past) - self.backward_epsilon)

    def decay_exploration_rate(self):
        """
        Decays the exploration rate, ensuring it doesn't fall below the minimum.
        """
        self.exploration_rate = max(self.min_exploration_rate, 
                                    self.exploration_rate * self.exploration_decay)

if __name__ == '__main__':
    # This part is for basic testing of the QPAgent itself.
    # A more comprehensive training script will be separate.
    print("Testing QPAgent basic functionalities...")

    # Mock environment
    class MockEnv:
        def __init__(self):
            self.action_space_size = 4 # 0, 1, 2, 3
            self.current_state = (0,0)

        def reset(self):
            self.current_state = (0,0)
            return self.current_state

        def step(self, action):
            # Mock step: move to a new state based on action, simple reward
            r, c = self.current_state
            if action == 0: # Up
                self.current_state = (max(0, r-1), c)
            elif action == 1: # Down
                self.current_state = (min(2, r+1), c) # Assume 3x3 grid for mock
            elif action == 2: # Left
                self.current_state = (r, max(0, c-1))
            elif action == 3: # Right
                self.current_state = (r, min(2, c+1))
            
            reward = -0.1
            done = False
            if self.current_state == (2,2): # Mock goal
                reward = 1.0
                done = True
            return self.current_state, reward, done, {}

    mock_env = MockEnv()
    agent = QPAgent(mock_env, alpha=0.5, backward_epsilon=0.05)

    state = mock_env.reset()
    print(f"Initial state: {state}")

    # Test choose_action (mostly exploration initially)
    action = agent.choose_action(state)
    print(f"Chosen action (exploration_rate={agent.exploration_rate:.2f}): {action}")
    assert 0 <= action < mock_env.action_space_size

    # Simulate a few steps and learning
    trajectory_history = []
    s = mock_env.reset()
    
    print("\nSimulating a few steps:")
    for i in range(5):
        a = agent.choose_action(s)
        s2, r, done, _ = mock_env.step(a)
        
        # Add current transition to trajectory *before* learning
        current_transition_info = (s, a, r)
        trajectory_history.append(current_transition_info)
        
        print(f"Step {i+1}: s={s}, a={a}, r={r}, s2={s2}")
        
        q_s_a_before = agent.q_table[s][a]
        # For backward update, find (sp, ap) if history exists
        q_sp_ap_before = None
        if len(trajectory_history) > 1:
            sp, ap, _ = trajectory_history[-2]
            q_sp_ap_before = agent.q_table[sp][ap]

        agent.learn(s, a, r, s2, trajectory_history)
        
        q_s_a_after = agent.q_table[s][a]
        print(f"  Forward update: q[{s}][{a}]: {q_s_a_before:.3f} -> {q_s_a_after:.3f}")

        if len(trajectory_history) > 1:
            sp, ap, _ = trajectory_history[-2]
            q_sp_ap_after = agent.q_table[sp][ap]
            print(f"  Backward update: q[{sp}][{ap}]: {q_sp_ap_before:.3f} -> {q_sp_ap_after:.3f}")
            # Check backward update tendency:
            # If abs(q_sp_ap_before) > agent.backward_epsilon, then abs(q_sp_ap_after) should be closer to agent.backward_epsilon
            # If abs(q_sp_ap_before) < agent.backward_epsilon, then abs(q_sp_ap_after) should be closer to agent.backward_epsilon
            # This is a bit complex to assert simply due to the sign.
            # Example: if q_sp_ap_before = 0.5, eps=0.05. abs(q)-eps = 0.45. q_after = 0.5 - alpha*0.45.
            # Example: if q_sp_ap_before = -0.5, eps=0.05. abs(q)-eps = 0.45. q_after = -0.5 - alpha*0.45.
            # Example: if q_sp_ap_before = 0.02, eps=0.05. abs(q)-eps = -0.03. q_after = 0.02 - alpha*(-0.03).
            # Example: if q_sp_ap_before = -0.02, eps=0.05. abs(q)-eps = -0.03. q_after = -0.02 - alpha*(-0.03).
            # So, if abs(q_val_past) > eps, q_val_past moves away from 0.
            # If abs(q_val_past) < eps, q_val_past moves towards 0.
            # This means it stabilizes at +/- eps.
            
        s = s2
        if done:
            print("Mock goal reached.")
            break
    
    agent.decay_exploration_rate()
    print(f"\nExploration rate after decay: {agent.exploration_rate:.4f}")

    print("\nQPAgent basic test finished.")
