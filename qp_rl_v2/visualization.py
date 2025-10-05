"""
Visualization utilities for Quasi-Probability RL experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Optional
import os

from .experiment import ExperimentResults
from .quasi_probability_agent import QuasiProbabilityAgent
from .grid_environment import GridWorld


def plot_learning_curves(
    qp_results: List[ExperimentResults],
    classical_results: List[ExperimentResults],
    window: int = 50,
    save_path: Optional[str] = None
):
    """
    Plot learning curves comparing QP-RL and classical Q-learning.

    Args:
        qp_results: Results from QP-RL agent
        classical_results: Results from classical agent
        window: Window size for smoothing
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot rewards
    ax1.set_title('Learning Curve: Rewards', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Average across runs
    qp_rewards = np.array([r.episode_rewards for r in qp_results])
    classical_rewards = np.array([r.episode_rewards for r in classical_results])

    # Smooth with moving average
    def smooth(data, window):
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        return smoothed

    episodes = np.arange(len(qp_rewards[0]))

    # Plot QP-RL
    qp_mean = np.mean(qp_rewards, axis=0)
    qp_std = np.std(qp_rewards, axis=0)
    qp_smooth = smooth(qp_mean, window)

    episodes_smooth = episodes[:len(qp_smooth)]
    ax1.plot(episodes_smooth, qp_smooth, label='QP-RL', color='blue', linewidth=2)
    ax1.fill_between(episodes_smooth,
                      smooth(qp_mean - qp_std, window),
                      smooth(qp_mean + qp_std, window),
                      alpha=0.2, color='blue')

    # Plot Classical
    classical_mean = np.mean(classical_rewards, axis=0)
    classical_std = np.std(classical_rewards, axis=0)
    classical_smooth = smooth(classical_mean, window)

    ax1.plot(episodes_smooth, classical_smooth, label='Classical Q-Learning',
             color='orange', linewidth=2)
    ax1.fill_between(episodes_smooth,
                      smooth(classical_mean - classical_std, window),
                      smooth(classical_mean + classical_std, window),
                      alpha=0.2, color='orange')

    ax1.legend(fontsize=11)

    # Plot steps
    ax2.set_title('Learning Curve: Steps to Goal', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps', fontsize=12)
    ax2.grid(True, alpha=0.3)

    qp_steps = np.array([r.episode_steps for r in qp_results])
    classical_steps = np.array([r.episode_steps for r in classical_results])

    qp_steps_mean = np.mean(qp_steps, axis=0)
    qp_steps_std = np.std(qp_steps, axis=0)
    qp_steps_smooth = smooth(qp_steps_mean, window)

    ax2.plot(episodes_smooth, qp_steps_smooth, label='QP-RL', color='blue', linewidth=2)
    ax2.fill_between(episodes_smooth,
                      smooth(qp_steps_mean - qp_steps_std, window),
                      smooth(qp_steps_mean + qp_steps_std, window),
                      alpha=0.2, color='blue')

    classical_steps_mean = np.mean(classical_steps, axis=0)
    classical_steps_std = np.std(classical_steps, axis=0)
    classical_steps_smooth = smooth(classical_steps_mean, window)

    ax2.plot(episodes_smooth, classical_steps_smooth, label='Classical Q-Learning',
             color='orange', linewidth=2)
    ax2.fill_between(episodes_smooth,
                      smooth(classical_steps_mean - classical_steps_std, window),
                      smooth(classical_steps_mean + classical_steps_std, window),
                      alpha=0.2, color='orange')

    ax2.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")

    plt.show()


def visualize_quasi_probabilities(
    agent: QuasiProbabilityAgent,
    env: GridWorld,
    save_path: Optional[str] = None
):
    """
    Visualize quasi-probability distribution over the grid.

    Shows positive (future) and negative (past) Q-values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create grids for positive and negative Q-values
    positive_grid = np.zeros((env.rows, env.cols))
    negative_grid = np.zeros((env.rows, env.cols))

    for state in range(env.n_states):
        row = state // env.cols
        col = state % env.cols
        pos = (row, col)

        # Skip walls
        if pos in env.walls:
            continue

        # Get max and min Q-values for this state
        q_values = agent.Q[state]
        max_q = np.max(q_values)
        min_q = np.min(q_values)

        positive_grid[row, col] = max(0, max_q)
        negative_grid[row, col] = min(0, min_q)

    # Plot positive (future) Q-values
    ax1 = axes[0]
    ax1.set_title('Future Actions (Positive Q-values)', fontsize=14, fontweight='bold')
    im1 = ax1.imshow(positive_grid, cmap='Blues', origin='upper')

    # Add walls and goal
    for row in range(env.rows):
        for col in range(env.cols):
            pos = (row, col)
            if pos in env.walls:
                ax1.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=True, color='black'))
            elif pos == env.goal_pos:
                ax1.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=True, color='gold', alpha=0.5))
            elif pos == env.start_pos:
                ax1.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=False,
                                       edgecolor='red', linewidth=3))

    ax1.set_xticks(range(env.cols))
    ax1.set_yticks(range(env.rows))
    ax1.grid(True, color='gray', linewidth=0.5)
    plt.colorbar(im1, ax=ax1, label='Max Q-value')

    # Plot negative (past) Q-values
    ax2 = axes[1]
    ax2.set_title('Past Actions (Negative Q-values)', fontsize=14, fontweight='bold')
    im2 = ax2.imshow(negative_grid, cmap='Greens_r', origin='upper')

    for row in range(env.rows):
        for col in range(env.cols):
            pos = (row, col)
            if pos in env.walls:
                ax2.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=True, color='black'))
            elif pos == env.goal_pos:
                ax2.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=True, color='gold', alpha=0.5))
            elif pos == env.start_pos:
                ax2.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=False,
                                       edgecolor='red', linewidth=3))

    ax2.set_xticks(range(env.cols))
    ax2.set_yticks(range(env.rows))
    ax2.grid(True, color='gray', linewidth=0.5)
    plt.colorbar(im2, ax=ax2, label='Min Q-value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved quasi-probability visualization to {save_path}")

    plt.show()


def visualize_policy(
    agent,
    env: GridWorld,
    title: str = "Learned Policy",
    save_path: Optional[str] = None
):
    """
    Visualize the learned policy as arrows on the grid.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Draw grid
    for row in range(env.rows):
        for col in range(env.cols):
            pos = (row, col)

            # Draw cell
            if pos in env.walls:
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=True, color='black'))
            elif pos in env.traps:
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=True, color='red', alpha=0.3))
            elif pos == env.goal_pos:
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=True, color='gold', alpha=0.5))
            elif pos == env.start_pos:
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=False,
                                      edgecolor='green', linewidth=3))

            # Draw policy arrow
            if pos not in env.walls and pos != env.goal_pos:
                state = env._pos_to_state(pos)
                action = agent.get_policy(state)

                # Arrow directions
                dx, dy = 0, 0
                if action == 0:  # Up
                    dy = -0.3
                elif action == 1:  # Down
                    dy = 0.3
                elif action == 2:  # Left
                    dx = -0.3
                elif action == 3:  # Right
                    dx = 0.3

                ax.arrow(col, row, dx, dy, head_width=0.2, head_length=0.15,
                        fc='blue', ec='blue', alpha=0.7)

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(True, color='gray', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved policy visualization to {save_path}")

    plt.show()


def plot_comparison_bars(
    qp_results: List[ExperimentResults],
    classical_results: List[ExperimentResults],
    save_path: Optional[str] = None
):
    """
    Create bar plots comparing key metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ('avg_reward', 'Average Reward', 0),
        ('avg_steps', 'Average Steps', 1),
        ('convergence_episode', 'Convergence Episode', 2)
    ]

    for metric_name, metric_label, ax_idx in metrics:
        ax = axes[ax_idx]

        qp_values = [getattr(r, metric_name) for r in qp_results]
        classical_values = [getattr(r, metric_name) for r in classical_results]

        # Filter out -1 values for convergence
        if metric_name == 'convergence_episode':
            qp_values = [v for v in qp_values if v != -1]
            classical_values = [v for v in classical_values if v != -1]

        if not qp_values or not classical_values:
            continue

        qp_mean = np.mean(qp_values)
        qp_std = np.std(qp_values)
        classical_mean = np.mean(classical_values)
        classical_std = np.std(classical_values)

        x = [0, 1]
        means = [qp_mean, classical_mean]
        stds = [qp_std, classical_std]
        labels = ['QP-RL', 'Classical']
        colors = ['blue', 'orange']

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(metric_label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison bars to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded.")
    print("Use plot_learning_curves(), visualize_quasi_probabilities(), etc.")
