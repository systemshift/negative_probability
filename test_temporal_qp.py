#!/usr/bin/env python3
"""
Test script for the Temporal Quasi-Probability RL framework.
Demonstrates how negative probabilities enable bidirectional time navigation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from qp_rl_project.temporal_nav_env import TemporalNavigationEnv
from qp_rl_project.temporal_qp_agent import TemporalQPAgent
from qp_rl_project.standard_q_agent import StandardQAgent

def visualize_quasi_probability_distribution(agent, env, episode, save_path):
    """Visualize the quasi-probability distribution."""
    if not agent.quasi_prob_distribution:
        return
        
    # Separate positive and negative probabilities
    positive_states = []
    positive_probs = []
    negative_states = []
    negative_probs = []
    
    for state, prob in agent.quasi_prob_distribution.items():
        x, y, vx, vy, t = state
        if prob > 0 and state != agent.current_state:
            positive_states.append((x, y))
            positive_probs.append(prob)
        elif prob < 0:
            negative_states.append((x, y))
            negative_probs.append(abs(prob))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot positive probabilities (future)
    ax1.set_title(f'Future States (Positive Probabilities) - Episode {episode}')
    ax1.set_xlim(-0.5, env.cols - 0.5)
    ax1.set_ylim(-0.5, env.rows - 0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # Plot grid
    for x in range(env.cols):
        for y in range(env.rows):
            if (x, y) == env.goal_pos:
                ax1.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                          fill=True, color='gold', alpha=0.5))
            elif (x, y) in env.obstacles:
                ax1.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                          fill=True, color='black', alpha=0.5))
    
    # Current position
    if agent.current_state:
        cx, cy, _, _, _ = agent.current_state
        ax1.scatter(cx, cy, s=200, c='red', marker='*', zorder=5, label='Current')
    
    # Future states
    if positive_states:
        xs, ys = zip(*positive_states)
        sizes = [p * 5000 for p in positive_probs]
        ax1.scatter(xs, ys, s=sizes, c='blue', alpha=0.6, label='Future')
    
    ax1.legend()
    ax1.invert_yaxis()
    
    # Plot negative probabilities (past)
    ax2.set_title(f'Past States (Negative Probabilities) - Episode {episode}')
    ax2.set_xlim(-0.5, env.cols - 0.5)
    ax2.set_ylim(-0.5, env.rows - 0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    
    # Plot grid
    for x in range(env.cols):
        for y in range(env.rows):
            if (x, y) == env.goal_pos:
                ax2.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                          fill=True, color='gold', alpha=0.5))
            elif (x, y) in env.obstacles:
                ax2.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                          fill=True, color='black', alpha=0.5))
    
    # Current position
    if agent.current_state:
        cx, cy, _, _, _ = agent.current_state
        ax2.scatter(cx, cy, s=200, c='red', marker='*', zorder=5, label='Current')
    
    # Past states
    if negative_states:
        xs, ys = zip(*negative_states)
        sizes = [p * 5000 for p in negative_probs]
        ax2.scatter(xs, ys, s=sizes, c='green', alpha=0.6, label='Past')
    
    ax2.legend()
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_temporal_qp_experiment(num_episodes=200, visualize_every=50):
    """Run experiment comparing Temporal QP agent with standard Q-learning."""
    
    print("=== Temporal Quasi-Probability RL Experiment ===\n")
    
    # Create environment
    env = TemporalNavigationEnv(
        grid_size=(8, 8),
        max_velocity=2.0,
        friction=0.15,
        goal_pos=(7, 7),
        obstacles=[(3, 3), (3, 4), (4, 3), (5, 5), (5, 6), (6, 5)]
    )
    
    # Create agents
    temporal_agent = TemporalQPAgent(
        env,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=0.5,
        exploration_decay=0.995,
        temporal_horizon=4,
        negative_prob_weight=0.3
    )
    
    standard_agent = StandardQAgent(
        env,
        alpha=0.1,
        gamma=0.95,
        exploration_rate=0.5,
        exploration_decay=0.995,
        use_softmax=True
    )
    
    # Results storage
    temporal_rewards = []
    standard_rewards = []
    temporal_steps = []
    standard_steps = []
    temporal_jumps = []
    
    # Create results directory
    os.makedirs("results/temporal_qp", exist_ok=True)
    
    # Training loop
    for episode in range(num_episodes):
        # Train Temporal QP agent
        state = env.reset()
        temporal_agent.reset(state)
        episode_reward = 0
        episode_steps = 0
        
        for step in range(200):  # Max steps per episode
            # Visualize quasi-probability distribution
            if episode % visualize_every == 0 and step == 0:
                visualize_quasi_probability_distribution(
                    temporal_agent, env, episode,
                    f"results/temporal_qp/quasi_prob_dist_ep{episode}.png"
                )
            
            # Choose action
            action_or_jump = temporal_agent.choose_action()
            
            if isinstance(action_or_jump, tuple) and action_or_jump[0] == "JUMP":
                # Handle backward jump
                _, target_state = action_or_jump
                
                # Set environment to target state
                env.state = target_state
                env.time_step = target_state[4]
                state = target_state
                temporal_agent.current_state = target_state
                
                # Update quasi-probability distribution after jump
                temporal_agent._update_quasi_probability_distribution()
                
                episode_steps += 1
                continue
            
            # Regular forward action
            action = action_or_jump
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            temporal_agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        temporal_rewards.append(episode_reward)
        temporal_steps.append(episode_steps)
        temporal_jumps.append(temporal_agent.backward_jumps)
        
        # Train standard agent
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(200):
            action = standard_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            standard_agent.learn(state, action, reward, next_state, done, None)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        standard_agent.decay_exploration_rate()
        standard_rewards.append(episode_reward)
        standard_steps.append(episode_steps)
        
        # Print progress
        if (episode + 1) % 20 == 0:
            avg_temporal_reward = np.mean(temporal_rewards[-20:])
            avg_standard_reward = np.mean(standard_rewards[-20:])
            avg_jumps = np.mean(temporal_jumps[-20:])
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Temporal QP: Avg Reward = {avg_temporal_reward:.2f}, "
                  f"Avg Jumps = {avg_jumps:.1f}")
            print(f"  Standard Q:  Avg Reward = {avg_standard_reward:.2f}")
            print()
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reward comparison
    window = 10
    temporal_smooth = np.convolve(temporal_rewards, np.ones(window)/window, mode='valid')
    standard_smooth = np.convolve(standard_rewards, np.ones(window)/window, mode='valid')
    
    ax1.plot(temporal_smooth, label='Temporal QP Agent', color='blue')
    ax1.plot(standard_smooth, label='Standard Q Agent', color='orange')
    ax1.set_title('Learning Performance Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Steps to goal
    temporal_steps_smooth = np.convolve(temporal_steps, np.ones(window)/window, mode='valid')
    standard_steps_smooth = np.convolve(standard_steps, np.ones(window)/window, mode='valid')
    
    ax2.plot(temporal_steps_smooth, label='Temporal QP Agent', color='blue')
    ax2.plot(standard_steps_smooth, label='Standard Q Agent', color='orange')
    ax2.set_title('Steps to Goal')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Backward jumps
    jumps_smooth = np.convolve(temporal_jumps, np.ones(window)/window, mode='valid')
    ax3.plot(jumps_smooth, color='green')
    ax3.set_title('Temporal QP Agent: Backward Jumps per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Jumps')
    ax3.grid(True, alpha=0.3)
    
    # Final statistics
    ax4.axis('off')
    stats_text = f"""Final Statistics (Last 50 episodes):

Temporal QP Agent:
  Avg Reward: {np.mean(temporal_rewards[-50:]):.2f}
  Avg Steps: {np.mean(temporal_steps[-50:]):.2f}
  Avg Jumps: {np.mean(temporal_jumps[-50:]):.2f}
  Total Jumps: {sum(temporal_jumps)}

Standard Q Agent:
  Avg Reward: {np.mean(standard_rewards[-50:]):.2f}
  Avg Steps: {np.mean(standard_steps[-50:]):.2f}

Improvement:
  Reward: {(np.mean(temporal_rewards[-50:]) - np.mean(standard_rewards[-50:])) / abs(np.mean(standard_rewards[-50:])) * 100:.1f}%
  Steps: {(np.mean(standard_steps[-50:]) - np.mean(temporal_steps[-50:])) / np.mean(standard_steps[-50:]) * 100:.1f}% reduction
"""
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("results/temporal_qp/experiment_results.png", dpi=150)
    plt.close()
    
    print("\nExperiment completed! Results saved to results/temporal_qp/")
    
    # Demonstrate a single episode with detailed output
    print("\n=== Demonstrating Single Episode ===")
    state = env.reset()
    temporal_agent.reset(state)
    
    print(f"Initial state: {state}")
    env.render()
    
    for step in range(30):
        action_or_jump = temporal_agent.choose_action()
        
        if isinstance(action_or_jump, tuple) and action_or_jump[0] == "JUMP":
            _, target_state = action_or_jump
            print(f"\nStep {step}: BACKWARD JUMP")
            print(f"  From: {state}")
            print(f"  To:   {target_state}")
            
            env.state = target_state
            env.time_step = target_state[4]
            state = target_state
            temporal_agent.current_state = target_state
            temporal_agent._update_quasi_probability_distribution()
            
            env.render()
        else:
            action = action_or_jump
            next_state, reward, done, info = env.step(action)
            
            print(f"\nStep {step}: Forward action {action}")
            print(f"  Reward: {reward:.3f}")
            
            temporal_agent.update(state, action, reward, next_state, done)
            state = next_state
            
            env.render()
            
            if done:
                print("\nGoal reached!")
                break
    
    # Show final quasi-probability distribution
    print("\nFinal Quasi-Probability Distribution (top 10):")
    sorted_probs = sorted(temporal_agent.quasi_prob_distribution.items(), 
                         key=lambda x: abs(x[1]), reverse=True)[:10]
    for state, prob in sorted_probs:
        sign = "+" if prob >= 0 else ""
        x, y, vx, vy, t = state
        print(f"  Pos({x},{y}) Vel({vx:.2f},{vy:.2f}) T={t}: {sign}{prob:.3f}")

if __name__ == "__main__":
    run_temporal_qp_experiment(num_episodes=200, visualize_every=50)
