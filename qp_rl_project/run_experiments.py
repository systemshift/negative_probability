import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
from collections import defaultdict

from qp_rl_project.timeline_grid_env import TimelineGridEnv
from qp_rl_project.qp_agent import QPAgent
from qp_rl_project.standard_q_agent import StandardQAgent
from qp_rl_project.random_restart_q_agent import RandomRestartQAgent
from qp_rl_project.trap_grid_env import create_trap_grid

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def run_experiments_for_envs():
    """
    Run experiments for both standard and trap grid environments.
    This function serves as the main entry point that runs comparative
    experiments for both environment types.
    """
    print("\n=== Running experiments on standard grid environment ===")
    run_comparative_experiments(
        num_episodes=1000, 
        num_runs=3,  # Reduced for quicker results
        render_greedy=False,
        env_type="standard", 
        results_subdir="standard"
    )
    
    print("\n=== Running experiments on trap grid environment ===")
    run_comparative_experiments(
        num_episodes=1500,  # More episodes for the more complex environment
        num_runs=3,  # Reduced for quicker results
        render_greedy=False,
        env_type="trap", 
        results_subdir="trap"
    )

def run_single_experiment(agent, env, num_episodes=1000, max_steps=200, render_greedy=True):
    """
    Run a training experiment for a single agent.
    
    Args:
        agent: The RL agent to train
        env: The environment to train in
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        render_greedy: Whether to render the greedy evaluation run
        
    Returns:
        dict: Experiment results with metrics
    """
    results = {
        'episode_rewards': [],
        'episode_steps': [],
        'episode_jumps': [],
        'states_visited': set(),
        'unique_states_per_episode': [],
        'success_count': 0,
        'final_greedy_reward': None,
        'final_greedy_steps': None,
        'final_greedy_succeeded': False,
        'training_time': 0
    }
    
    start_time = time.time()
    
    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        current_trajectory = []
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_jumps = 0
        episode_states = set([state])  # Track unique states visited in this episode
        
        for step in range(max_steps):
            action_or_jump = agent.choose_action(state, current_trajectory)
            
            if isinstance(action_or_jump, tuple) and action_or_jump[0] == "JUMP":
                _, jumped_to_state = action_or_jump
                state = jumped_to_state
                episode_jumps += 1
                
                # Handle trajectory after jump (truncate to the state we jumped to)
                found_idx = -1
                for idx, (s_hist, _, _) in enumerate(current_trajectory):
                    if s_hist == jumped_to_state:
                        found_idx = idx
                        break
                if found_idx != -1:
                    current_trajectory = current_trajectory[:found_idx]
                
                # Add the jumped-to state to our tracked states
                episode_states.add(state)
                results['states_visited'].add(state)
                
                # No environment step for a jump
                continue
            
            # Regular forward action
            action = action_or_jump
            next_state, reward, done, _ = env.step(action)
            
            current_trajectory.append((state, action, reward))
            agent.learn(state, action, reward, next_state, done, current_trajectory)
            
            episode_reward += reward
            episode_states.add(next_state)
            results['states_visited'].add(next_state)
            
            state = next_state
            episode_steps += 1
            
            if done:
                results['success_count'] += 1
                break
        
        agent.decay_exploration_rate()
        results['episode_rewards'].append(episode_reward)
        results['episode_steps'].append(episode_steps)
        results['episode_jumps'].append(episode_jumps)
        results['unique_states_per_episode'].append(len(episode_states))
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Steps: {episode_steps}, "
                  f"Jumps: {episode_jumps}, "
                  f"Exp Rate: {agent.exploration_rate:.3f}")
    
    results['training_time'] = time.time() - start_time
    
    # Run a final greedy evaluation
    if render_greedy:
        print("\nRunning final greedy evaluation:")
    
    state = env.reset()
    if render_greedy:
        env.render()
    
    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = 0  # Force greedy behavior
    
    greedy_reward = 0
    greedy_steps = 0
    greedy_success = False
    
    for _ in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        if render_greedy:
            print(f"Greedy move: From {state} action {action} -> {next_state}, R: {reward}")
            env.render()
        
        greedy_reward += reward
        greedy_steps += 1
        state = next_state
        
        if done:
            greedy_success = True
            if render_greedy:
                print("\nGreedy evaluation: Goal reached!")
            break
    
    if render_greedy and not greedy_success:
        print("\nGreedy evaluation: Failed to reach goal within step limit.")
    
    print(f"Greedy evaluation: Reward: {greedy_reward:.2f}, Steps: {greedy_steps}, Success: {greedy_success}")
    
    # Restore exploration rate
    agent.exploration_rate = original_exploration_rate
    
    # Record greedy evaluation results
    results['final_greedy_reward'] = greedy_reward
    results['final_greedy_steps'] = greedy_steps
    results['final_greedy_succeeded'] = greedy_success
    
    return results

def create_env(env_type="standard", config=None):
    """
    Create a grid environment based on the specified type.
    
    Args:
        env_type (str): Type of environment to create - "standard" or "trap"
        config (dict): Optional custom configuration (only used for "standard" type)
        
    Returns:
        TimelineGridEnv: The created environment
    """
    if env_type == "trap":
        return create_trap_grid()
    else:  # standard
        if config is None:
            config = {
                "grid_size": (5, 5),
                "start_pos": (0, 0),
                "goal_pos": (4, 4),
                "walls": [(1,1), (1,2), (1,3), (3,1), (3,2), (3,3)],
                "one_way_doors": [{'from': (0,2), 'to': (2,2), 'action': 1}],  # Down from (0,2) to (2,2)
                "portals": {(2,4): (4,0)}  # Portal from (2,4) to (4,0)
            }
        return TimelineGridEnv(**config)

def plot_comparison(results_dict, metric='episode_rewards', window_size=10, 
                    title='Reward Comparison', ylabel='Reward', filename='comparison.png',
                    results_subdir="standard"):
    """
    Plot a comparison of a metric across different agents.
    
    Args:
        results_dict: Dictionary of agent_name -> results from run_single_experiment
        metric: The metric to plot
        window_size: Window size for the moving average
        title, ylabel, filename: Plot formatting options
    """
    plt.figure(figsize=(12, 8))
    
    for agent_name, results in results_dict.items():
        data = results[metric]
        x = range(len(data))
        
        # Plot raw data with low alpha
        plt.plot(x, data, alpha=0.3, label=f"{agent_name} (raw)")
        
        # Calculate and plot moving average
        if len(data) >= window_size:
            moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(data)), moving_avg, 
                     label=f"{agent_name} (MA-{window_size})")
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    results_path = os.path.join('results', results_subdir)
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(os.path.join(results_path, filename))
    print(f"Plot saved to {results_path}/{filename}")
    plt.close()

def plot_jump_heatmap(agent_name, q_agent, grid_size, filename='q_heatmap.png', results_subdir="standard"):
    """
    Create a heatmap visualization of Q-values, with positive values in red and negative in blue.
    
    Args:
        agent_name: Name of the agent (for title)
        q_agent: Agent with a q_table attribute
        grid_size: Size of the grid (rows, cols)
        filename: Filename to save the plot
    """
    rows, cols = grid_size
    
    # Create action-specific matrices for visualization
    action_names = ['Up', 'Down', 'Left', 'Right']
    q_matrices = {a_name: np.zeros((rows, cols)) for a_name in action_names}
    max_q_matrix = np.zeros((rows, cols))
    min_q_matrix = np.zeros((rows, cols))
    
    # Fill matrices with Q-values
    for state, actions in q_agent.q_table.items():
        if len(state) == 2:  # Make sure state is a valid (row, col) tuple
            r, c = state
            if 0 <= r < rows and 0 <= c < cols:
                # Record min/max Q-values for this state
                if actions:
                    q_values = list(actions.values())
                    max_q_matrix[r, c] = max(q_values)
                    min_q_matrix[r, c] = min(q_values)
                    
                    # Record action-specific Q-values
                    for action_idx, action_name in enumerate(action_names):
                        if action_idx in actions:
                            q_matrices[action_name][r, c] = actions[action_idx]
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Q-value Visualization for {agent_name}', fontsize=16)
    
    # Plot action-specific heatmaps
    for i, (action_name, q_matrix) in enumerate(q_matrices.items()):
        row, col = i // 2, i % 2
        im = axs[row, col].imshow(q_matrix, cmap='coolwarm', interpolation='nearest')
        axs[row, col].set_title(f'Q-values for {action_name} Action')
        axs[row, col].set_xticks(range(cols))
        axs[row, col].set_yticks(range(rows))
        fig.colorbar(im, ax=axs[row, col])
        
        # Annotate with actual values
        for r in range(rows):
            for c in range(cols):
                if q_matrix[r, c] != 0:  # Only annotate non-zero values
                    axs[row, col].text(c, r, f'{q_matrix[r, c]:.2f}', 
                                      ha="center", va="center", color="black",
                                      fontsize=8)
    
    # Plot max Q-value heatmap
    im = axs[1, 2].imshow(max_q_matrix, cmap='hot', interpolation='nearest')
    axs[1, 2].set_title('Max Q-value per State')
    axs[1, 2].set_xticks(range(cols))
    axs[1, 2].set_yticks(range(rows))
    fig.colorbar(im, ax=axs[1, 2])
    
    # Annotate with actual values
    for r in range(rows):
        for c in range(cols):
            if max_q_matrix[r, c] != 0:
                axs[1, 2].text(c, r, f'{max_q_matrix[r, c]:.2f}', 
                              ha="center", va="center", color="black",
                              fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    results_path = os.path.join('results', results_subdir)
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(os.path.join(results_path, filename))
    print(f"Q-value heatmap saved to {results_path}/{filename}")
    plt.close()

def run_comparative_experiments(num_episodes=1000, num_runs=5, render_greedy=False, 
                               env_type="standard", results_subdir="standard"):
    """
    Run experiments for multiple agent types and compare their performance.
    
    Args:
        num_episodes: Number of episodes per experiment
        num_runs: Number of runs with different random seeds for statistical significance
        render_greedy: Whether to render the final greedy evaluation
    """
    # Seed for reproducibility, but we'll vary it across runs
    base_seed = 42
    
    # Default parameters for all agents
    base_params = {
        "alpha": 0.1,
        "gamma": 0.9,
        "exploration_rate": 1.0,
        "exploration_decay": 0.999,
        "min_exploration_rate": 0.01,
        "q_init_val": 0.0
    }
    
    # Agent configurations
    agent_configs = {
        "QPAgent": {
            **base_params,
            "backward_epsilon": 0.01,
            "softmax_temp": 0.5,
            "min_trajectory_for_jump": 3
        },
        "StandardQAgent_eps": {
            **base_params,
            "use_softmax": False
        },
        "StandardQAgent_softmax": {
            **base_params,
            "use_softmax": True,
            "softmax_temp": 0.5
        },
        "RandomRestartQAgent": {
            **base_params,
            "use_softmax": False,
            "restart_probability": 0.1,
            "restart_threshold": 5
        }
    }
    
    # Prepare to collect results
    all_results = defaultdict(list)
    aggregated_metrics = defaultdict(lambda: defaultdict(list))
    
    for run in range(num_runs):
        run_seed = base_seed + run
        print(f"\n=== Starting Run {run+1}/{num_runs} (seed: {run_seed}) ===")
        np.random.seed(run_seed)
        random.seed(run_seed)
        
        # Create an environment for this run
        env = create_env(env_type=env_type)
        
        # Train each agent type
        for agent_name, agent_config in agent_configs.items():
            print(f"\n--- Training {agent_name} ---")
            
            # Create the appropriate agent type
            if agent_name == "QPAgent":
                agent = QPAgent(env, **agent_config)
            elif agent_name.startswith("StandardQAgent"):
                agent = StandardQAgent(env, **agent_config)
            elif agent_name == "RandomRestartQAgent":
                agent = RandomRestartQAgent(env, **agent_config)
            else:
                raise ValueError(f"Unknown agent type: {agent_name}")
            
            # Run the experiment
            results = run_single_experiment(
                agent, 
                env, 
                num_episodes=num_episodes, 
                render_greedy=(render_greedy and run == 0)  # Only render for first run
            )
            
            # Save results
            all_results[f"{agent_name}_run{run}"] = results
            
            # Collect metrics for aggregation
            for metric in ['episode_rewards', 'episode_steps', 'episode_jumps', 'unique_states_per_episode']:
                aggregated_metrics[agent_name][metric].append(results[metric])
            
            aggregated_metrics[agent_name]['final_greedy_reward'].append(results['final_greedy_reward'])
            aggregated_metrics[agent_name]['final_greedy_steps'].append(results['final_greedy_steps'])
            aggregated_metrics[agent_name]['final_greedy_succeeded'].append(results['final_greedy_succeeded'])
            aggregated_metrics[agent_name]['training_time'].append(results['training_time'])
            aggregated_metrics[agent_name]['success_count'].append(results['success_count'])
            aggregated_metrics[agent_name]['unique_states_total'].append(len(results['states_visited']))
            
            # Create Q-value heatmap for this agent
            if run == 0:  # Only for the first run to avoid too many plots
                plot_jump_heatmap(agent_name, agent, env.grid_size, f"{agent_name}_q_heatmap.png")
        
        # Generate run-specific plots
        plot_comparison(
            {k: all_results[f"{k}_run{run}"] for k in agent_configs.keys()},
            metric='episode_rewards',
            title=f'Reward Comparison (Run {run+1})',
            filename=f'reward_comparison_run{run+1}.png'
        )
        
        plot_comparison(
            {k: all_results[f"{k}_run{run}"] for k in agent_configs.keys()},
            metric='episode_jumps',
            title=f'Jump Count Comparison (Run {run+1})',
            ylabel='Number of Jumps',
            filename=f'jump_comparison_run{run+1}.png'
        )
        
        plot_comparison(
            {k: all_results[f"{k}_run{run}"] for k in agent_configs.keys()},
            metric='unique_states_per_episode',
            title=f'Exploration Comparison (Run {run+1})',
            ylabel='Unique States Visited',
            filename=f'exploration_comparison_run{run+1}.png'
        )
    
    # Save all results
    with open(os.path.join('results', 'experiment_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    # Calculate and print aggregate statistics
    print("\n=== Aggregate Results ===")
    
    for agent_name in agent_configs.keys():
        print(f"\n{agent_name}:")
        
        # Calculate means and standard deviations for key metrics
        for metric in ['final_greedy_reward', 'final_greedy_steps', 'training_time', 
                       'success_count', 'unique_states_total']:
            values = aggregated_metrics[agent_name][metric]
            print(f"  {metric}: Mean = {np.mean(values):.2f}, StdDev = {np.std(values):.2f}")
        
        success_rate = np.mean(aggregated_metrics[agent_name]['final_greedy_succeeded']) * 100
        print(f"  Greedy Success Rate: {success_rate:.1f}%")
    
    # Create final aggregate plots with error bands for variance across runs
    for metric, ylabel in [
        ('episode_rewards', 'Reward'),
        ('episode_steps', 'Steps'),
        ('episode_jumps', 'Number of Jumps'),
        ('unique_states_per_episode', 'Unique States Visited')
    ]:
        plt.figure(figsize=(12, 8))
        
        for agent_name in agent_configs.keys():
            # Get data arrays from all runs
            data_arrays = aggregated_metrics[agent_name][metric]
            max_len = max(len(arr) for arr in data_arrays)
            
            # Pad arrays to equal length
            padded_arrays = []
            for arr in data_arrays:
                if len(arr) < max_len:
                    padded = np.pad(arr, (0, max_len - len(arr)), 'edge')
                else:
                    padded = np.array(arr)
                padded_arrays.append(padded)
            
            # Convert to numpy array for easier calculations
            data = np.array(padded_arrays)
            
            # Calculate mean and std for each episode across runs
            mean_data = np.mean(data, axis=0)
            std_data = np.std(data, axis=0)
            
            # Apply smoothing to mean and std
            window_size = 10
            if len(mean_data) >= window_size:
                mean_smooth = np.convolve(mean_data, np.ones(window_size)/window_size, mode='valid')
                std_smooth = np.convolve(std_data, np.ones(window_size)/window_size, mode='valid')
                x = range(window_size-1, len(mean_data))
            else:
                mean_smooth = mean_data
                std_smooth = std_data
                x = range(len(mean_data))
            
            # Plot mean with error band
            plt.plot(x, mean_smooth, label=agent_name)
            plt.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.2)
        
        plt.title(f'Average {metric.replace("_", " ").title()} Across {num_runs} Runs')
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join('results', f'aggregate_{metric}.png'))
        print(f"Aggregate plot saved to results/aggregate_{metric}.png")
        plt.close()

if __name__ == "__main__":
    # Ensure random module is imported
    import random
    
    print("Starting comparative experiments...")
    run_experiments_for_envs()
    print("Experiments completed. Results saved in the 'results' directory.")
