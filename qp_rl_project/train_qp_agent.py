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
        "backward_epsilon": 0.01    # eps for backward update
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
        
        for step in range(max_steps_per_episode):
            action = agent.choose_action(state, current_episode_trajectory) # Pass trajectory (though not used by basic choose_action yet)
            
            next_state, reward, done, _ = env.step(action)
            
            # Store transition info for learning
            # The agent.learn method expects (s,a,r) of current transition to be in trajectory
            current_episode_trajectory.append((state, action, reward))
            
            agent.learn(state, action, reward, next_state, current_episode_trajectory)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_exploration_rate()
        all_episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} finished. Reward: {episode_reward:.2f}, Exploration Rate: {agent.exploration_rate:.3f}")
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
