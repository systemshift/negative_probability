#!/usr/bin/env python3
"""
Test script for the Quasi-Probability Reinforcement Learning package.
This script runs a simple experiment to verify that the package is working correctly.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from qp_rl_project.trap_grid_env import create_trap_grid
from qp_rl_project.qp_agent import QPAgent
from qp_rl_project.standard_q_agent import StandardQAgent

def run_quick_test(num_episodes=100):
    """Run a quick test of the QP-RL framework."""
    print("Creating trap grid environment...")
    env = create_trap_grid()
    env.reset()
    print("Trap Grid Environment:")
    env.render()
    
    print("\nInitializing QP-RL agent...")
    qp_agent = QPAgent(
        env,
        alpha=0.1,
        gamma=0.9,
        exploration_rate=1.0,
        exploration_decay=0.99,
        backward_epsilon=0.01,
        softmax_temp=0.5,
        min_trajectory_for_jump=3
    )
    
    print("Initializing standard Q-learning agent...")
    std_agent = StandardQAgent(
        env,
        alpha=0.1,
        gamma=0.9,
        exploration_rate=1.0,
        exploration_decay=0.99,
        use_softmax=False
    )
    
    # Train both agents for a few episodes
    print(f"\nTraining both agents for {num_episodes} episodes...")
    qp_rewards = []
    std_rewards = []
    qp_jumps = []
    
    for episode in range(num_episodes):
        # Train QP agent
        qp_trajectory = []
        state = env.reset()
        episode_reward = 0
        episode_jumps = 0
        
        for _ in range(100):  # Max steps per episode
            action_or_jump = qp_agent.choose_action(state, qp_trajectory)
            
            if isinstance(action_or_jump, tuple) and action_or_jump[0] == "JUMP":
                _, jumped_to_state = action_or_jump
                state = jumped_to_state
                episode_jumps += 1
                
                # Handle trajectory after jump
                found_idx = -1
                for idx, (s_hist, _, _) in enumerate(qp_trajectory):
                    if s_hist == jumped_to_state:
                        found_idx = idx
                        break
                if found_idx != -1:
                    qp_trajectory = qp_trajectory[:found_idx]
                
                continue
            
            # Regular forward action
            action = action_or_jump
            next_state, reward, done, _ = env.step(action)
            
            qp_trajectory.append((state, action, reward))
            qp_agent.learn(state, action, reward, next_state, done, qp_trajectory)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        qp_agent.decay_exploration_rate()
        qp_rewards.append(episode_reward)
        qp_jumps.append(episode_jumps)
        
        # Train standard agent
        state = env.reset()
        episode_reward = 0
        
        for _ in range(100):  # Max steps per episode
            action = std_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            std_agent.learn(state, action, reward, next_state, done, None)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        std_agent.decay_exploration_rate()
        std_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"QP Reward = {qp_rewards[-1]:.2f} (jumps: {qp_jumps[-1]}), "
                  f"Std Reward = {std_rewards[-1]:.2f}")
    
    # Plot results
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(qp_rewards, label='QP-RL Agent')
    plt.plot(std_rewards, label='Standard Q Agent')
    plt.title('Reward Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/quick_test_comparison.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(qp_jumps)
    plt.title('QP-RL Agent Jump Count')
    plt.xlabel('Episode')
    plt.ylabel('Number of Jumps')
    plt.grid(True)
    plt.savefig("results/quick_test_jumps.png")
    plt.close()
    
    print("\nTest completed. Results saved to the 'results' directory.")
    print(f"Final QP-RL agent exploration rate: {qp_agent.exploration_rate:.3f}")
    print(f"Final Standard agent exploration rate: {std_agent.exploration_rate:.3f}")

if __name__ == "__main__":
    run_quick_test(100)
