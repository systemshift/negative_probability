import collections
import random
import numpy as np

class QPAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 min_exploration_rate=0.01, q_init_val=0.0, 
                 backward_epsilon=0.01, softmax_temp=1.0,
                 min_trajectory_for_jump=2): # p_backward_jump removed
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
            softmax_temp (float): Temperature parameter for softmax action selection.
                                  Higher temp = more randomness, Lower temp = more greedy.
            min_trajectory_for_jump (int): Minimum length of trajectory_history to consider a jump.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_init_val = q_init_val
        self.backward_epsilon = backward_epsilon
        self.softmax_temp = softmax_temp
        # self.p_backward_jump = p_backward_jump # No longer used
        self.min_trajectory_for_jump = min_trajectory_for_jump

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
            # Explore: choose a random environment action
            return random.randrange(self.action_space_size)
        else:
            # Exploit: Implement "sample_qp" logic more directly.
            # 1. Sample an action `a_sampled` using softmax over q_table[state].
            # 2. Check the sign of q_table[state][a_sampled].
            #    - If positive (or non-negative), take action `a_sampled` forward.
            #    - If negative, trigger a backward jump.
            
            state_q_values_dict = self.q_table[state]
            if not state_q_values_dict: 
                return random.randrange(self.action_space_size)

            actions = list(state_q_values_dict.keys())
            q_values = np.array([state_q_values_dict[a] for a in actions])

            sampled_action_idx = -1
            if self.softmax_temp <= 0: # Greedy selection
                max_q = np.max(q_values)
                best_action_indices = [i for i, q_val in enumerate(q_values) if q_val == max_q]
                if not best_action_indices: # Should not happen if actions is not empty
                    return random.randrange(self.action_space_size)
                sampled_action_idx = random.choice(best_action_indices)
            else: # Softmax selection
                q_values_stable = q_values - np.max(q_values) 
                exp_q_values = np.exp(q_values_stable / self.softmax_temp)
                probs = exp_q_values / np.sum(exp_q_values)
                
                if np.isclose(np.sum(probs), 0.0) or not np.all(np.isfinite(probs)) or len(actions)==0:
                    return random.randrange(self.action_space_size) # Fallback
                
                # Choose an index based on probabilities
                sampled_action_idx = np.random.choice(len(actions), p=probs)

            if sampled_action_idx == -1: # Should not be reached if logic above is correct
                 return random.randrange(self.action_space_size)

            a_sampled = actions[sampled_action_idx]
            q_original_sampled = state_q_values_dict[a_sampled]

            # "Positive draws move forward; negative draws trigger a jump"
            # Let's define "positive draw" as q_original_sampled >= 0 (or a small positive threshold like self.backward_epsilon)
            # and "negative draw" as q_original_sampled < 0 (or below -self.backward_epsilon)
            
            # For now, simple >= 0 check for forward.
            if q_original_sampled >= 0:
                return a_sampled # This is an environment action for a forward step
            else:
                # Negative draw: Trigger a jump.
                # The jump target selection logic can be the same as before.
                if trajectory_history and len(trajectory_history) >= self.min_trajectory_for_jump:
                    candidate_jumps = []
                    for i in range(len(trajectory_history) -1): 
                        past_s, past_a, _ = trajectory_history[i]
                        q_val_past_sa = self.q_table[past_s][past_a]
                        if q_val_past_sa < 0:
                            score = abs(q_val_past_sa - (-self.backward_epsilon))
                            candidate_jumps.append({'state_to_jump_to': past_s, 'score': score})
                    
                    if candidate_jumps:
                        candidate_jumps.sort(key=lambda x: x['score'])
                        best_jump_target_state = candidate_jumps[0]['state_to_jump_to']
                        return ("JUMP", best_jump_target_state)
                
                # If jump cannot be made (e.g., no suitable past states or short trajectory),
                # or if q_original_sampled was negative but no jump candidates found,
                # then fall back to taking the sampled action `a_sampled` forward.
                # This means even if q(s,a) is negative, we might still take it as a forward step.
                # This part needs careful thought: if a "negative draw" *must* be a jump,
                # and no jump is possible, what happens? Explore?
                # For now, if jump fails, proceed with a_sampled.
                return a_sampled

    def learn(self, s, a, r, s2, done_s2, trajectory):
        """
        Updates the Q-table based on the transition and trajectory.
        Implements both forward (future) and backward (past) updates.

        Args:
            s: Current state.
            a: Action taken.
            r: Reward received.
            s2: Next state.
            done_s2 (bool): True if s2 is a terminal state, False otherwise.
            trajectory (list): History of (state, action, reward) tuples for the current episode.
                               Assumes `(s, a, r)` of the current transition has ALREADY been added.
        """
        # Forward Update (Future) - Standard Q-learning update
        current_q_sa = self.q_table[s][a]
        
        max_next_q = 0.0
        if not done_s2: # Only consider future rewards if s2 is not terminal
            if self.q_table[s2]: # If s2 has been visited and has Q-values
                max_next_q = max(self.q_table[s2].values())
            # If s2 is new (not in q_table), max_next_q remains 0.0 (optimistic init or q_init_val if used)
        
        target_q_sa = r + self.gamma * max_next_q
        self.q_table[s][a] = current_q_sa + self.alpha * (target_q_sa - current_q_sa)

        # Backward Update (Past)
        if len(trajectory) > 1: 
            sp, ap, _ = trajectory[-2] 
            q_val_past = self.q_table[sp][ap]
            self.q_table[sp][ap] = q_val_past - self.alpha * (q_val_past + self.backward_epsilon)

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
