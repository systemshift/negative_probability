import matplotlib.pyplot as plt
import numpy as np

from timeline_grid_env import TimelineGridEnv # Assuming it's in the same directory or PYTHONPATH
from qp_agent import QPAgent

def plot_rewards(episode_rewards, filename="qp_rl_episode_rewards.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Calculate and plot a simple moving average
    if len(episode_rewards) >= 10:
        moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        plt.plot(np.arange(9, len(episode_rewards)), moving_avg, label='10-episode SMA', color='red')
        plt.legend()
        
    plt.savefig(filename)
    print(f"Reward plot saved to {filename}")
    plt.close()

def main():
    # --- Environment Configuration ---
    # You can define different environments here to test the agent
    env_config_default = {
        "grid_size": (5, 5),
        "start_pos": (0, 0),
        "goal_pos": (4, 4),
        "walls": [(1,1), (1,2), (1,3), (3,1), (3,2), (3,3)],
        "one_way_doors": [{'from': (0,2), 'to': (2,2), 'action': 1}], # Down from (0,2) to (2,2)
        "portals": {(2,4): (4,0)} # Portal from (2,4) to (4,0)
    }
    
    env = TimelineGridEnv(**env_config_default)

    # --- Agent Configuration ---
    agent_config = {
        "alpha": 0.1,               # Learning rate
        "gamma": 0.9,               # Discount factor (currently not used in simple forward update)
        "exploration_rate": 1.0,    # Initial exploration rate
        "exploration_decay": 0.999, # Slower decay for more episodes
        "min_exploration_rate": 0.01,
        "q_init_val": 0.0,
        "backward_epsilon": 0.01,   # eps for backward update
        "softmax_temp": 0.5,        # Temperature for softmax action selection
        "p_backward_jump": 0.1,     # Probability of attempting a backward jump
        "min_trajectory_for_jump": 3 # Min trajectory length to consider a jump (e.g. S0->S1->S2, now at S2, can jump to S0)
    }
    agent = QPAgent(env, **agent_config)

    # --- Training Parameters ---
    num_episodes = 1000
    max_steps_per_episode = 200 # Prevent infinitely long episodes

    all_episode_rewards = []

    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        current_episode_trajectory = []
        state = env.reset()
        episode_reward = 0
        num_jumps_this_episode = 0
        
        for step_num in range(max_steps_per_episode): # Renamed step to step_num to avoid conflict
            action_or_jump_info = agent.choose_action(state, current_episode_trajectory)
            
            if isinstance(action_or_jump_info, tuple) and action_or_jump_info[0] == "JUMP":
                _, jumped_to_state = action_or_jump_info
                # print(f"Episode {episode+1}, Step {step_num+1}: Jumped from {state} to {jumped_to_state}")
                state = jumped_to_state
                num_jumps_this_episode += 1
                
                # Truncate trajectory: find the first occurrence of jumped_to_state and keep history up to that point.
                # This means we are rewinding to the point *after* jumped_to_state was first experienced via an action.
                # The state in trajectory is s, so we look for trajectory entries where s == jumped_to_state
                # More accurately, we should find the index of the (s,a,r) tuple where s was jumped_to_state
                # and the agent is now about to choose an action from jumped_to_state.
                # So, the trajectory should reflect the path taken to reach jumped_to_state.
                
                # Find the index of the first time 'jumped_to_state' was the *resulting state* s2 of a transition (s,a,r) -> s2
                # This is complex. A simpler trajectory management for now:
                # Find the last index where jumped_to_state was the 's' component.
                # This means we are rewinding to the point where an action was taken *from* jumped_to_state.
                
                # Simpler: Truncate trajectory to the point where jumped_to_state was the *start* of a recorded (s,a,r)
                # This means we are rewinding as if we just arrived at jumped_to_state and are about to act from it.
                # The history *leading to* jumped_to_state should be preserved.
                
                # Let's find the first index i where trajectory_history[i][0] == jumped_to_state
                # This means we are rewinding to the first time we were *in* jumped_to_state and about to take an action.
                # The trajectory should then be sliced up to, but not including, that point,
                # as we are now *at* jumped_to_state, and the history is what led *before* it.
                # This is still tricky.
                
                # Simplest for now: Truncate to the entry *before* the first time jumped_to_state was visited as 's'.
                # Or, if jumped_to_state was s in trajectory[k] = (s,a,r), then new trajectory is trajectory[:k]
                # This means the history is up to the point *before* we first acted from jumped_to_state.
                
                # Let's try this: find the first entry (s_hist, a_hist, r_hist) where s_hist == jumped_to_state.
                # The new trajectory will be all entries *before* this one.
                # This means we are truly rewinding to a point *before* we first acted from jumped_to_state.
                
                # If trajectory_history = [(s0,a0,r0), (s1,a1,r1), (s2,a2,r2)] and we jump to s1.
                # New trajectory should be [(s0,a0,r0)]. State is s1.
                
                found_idx = -1
                for idx, (s_hist, _, _) in enumerate(current_episode_trajectory):
                    if s_hist == jumped_to_state:
                        found_idx = idx
                        break
                if found_idx != -1:
                    current_episode_trajectory = current_episode_trajectory[:found_idx]
                # If jumped_to_state was the very first state (e.g. env.reset()), trajectory becomes empty.
                
                # No env.step(), no reward from env, no learning call for the jump itself.
                # done status doesn't change by jump itself.
                if done: # If already done and we jump, break (should not happen if goal is terminal)
                    break 
                continue # Continue to next step in the episode from the new state

            # Standard forward action
            action = action_or_jump_info
            next_state, reward, done, _ = env.step(action) # 'done' here is for next_state
            
            current_episode_trajectory.append((state, action, reward))
            agent.learn(state, action, reward, next_state, done, current_episode_trajectory) # Pass 'done' as done_s2
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_exploration_rate()
        all_episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} finished. Reward: {episode_reward:.2f}, Jumps: {num_jumps_this_episode}, Exp Rate: {agent.exploration_rate:.3f}")
            # Optional: Render the last state of an episode
            # print("Final state of episode:")
            # env.render()

    print("Training finished.")
    
    # Plotting rewards
    plot_rewards(all_episode_rewards)

    # You could add code here to inspect the Q-table or run the agent in a test mode without exploration.
    # For example, print some Q-values:
    print("\nSample Q-values (first 5 states with entries):")
    count = 0
    for s_key, actions_dict in agent.q_table.items():
        if count < 5:
            print(f"State {s_key}:")
            for a_key, q_val in actions_dict.items():
                print(f"  Action {a_key}: {q_val:.3f}")
            count += 1
        else:
            break
    
    # Example of running the trained agent greedily for one episode
    print("\nRunning trained agent (greedy):")
    state = env.reset()
    env.render()
    episode_reward_greedy = 0
    original_exploration_rate = agent.exploration_rate # save it
    agent.exploration_rate = 0 # greedy
    for _ in range(max_steps_per_episode):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        print(f"Greedy move: From {state} action {action} -> {next_state}, R: {reward}")
        env.render()
        episode_reward_greedy += reward
        state = next_state
        if done:
            print("Greedy run finished - Goal Reached!")
            break
    print(f"Total reward (greedy run): {episode_reward_greedy}")
    agent.exploration_rate = original_exploration_rate # restore it

if __name__ == '__main__':
    main()
