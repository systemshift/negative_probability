#!/usr/bin/env python3
"""
Main experiment script for Quasi-Probability RL

Runs comprehensive experiments comparing QP-RL with classical Q-learning.
"""

import os
import argparse
import numpy as np

from qp_rl_v2.grid_environment import (
    create_trap_maze,
    create_long_corridor,
    create_four_rooms
)
from qp_rl_v2.experiment import (
    compare_agents,
    print_comparison_summary
)
from qp_rl_v2.visualization import (
    plot_learning_curves,
    visualize_quasi_probabilities,
    visualize_policy,
    plot_comparison_bars
)
from qp_rl_v2.quasi_probability_agent import QuasiProbabilityAgent, ClassicalQAgent


def run_full_experiment(
    env_name: str = "trap_maze",
    n_episodes: int = 1000,
    n_runs: int = 5,
    output_dir: str = "results_v2"
):
    """
    Run full experiment suite on a given environment.

    Args:
        env_name: Name of environment ('trap_maze', 'long_corridor', 'four_rooms')
        n_episodes: Number of training episodes per run
        n_runs: Number of independent runs
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print(f"QUASI-PROBABILITY RL EXPERIMENT")
    print("="*70)
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Runs: {n_runs}")
    print("="*70 + "\n")

    # Create environment
    if env_name == "trap_maze":
        env = create_trap_maze()
        env_display_name = "Trap Maze (5x5)"
    elif env_name == "long_corridor":
        env = create_long_corridor()
        env_display_name = "Long Corridor (3x7)"
    elif env_name == "four_rooms":
        env = create_four_rooms()
        env_display_name = "Four Rooms (9x9)"
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    print(f"\nEnvironment: {env_display_name}")
    env.render()

    # Create output directory
    env_output_dir = os.path.join(output_dir, env_name)
    os.makedirs(env_output_dir, exist_ok=True)

    # Run comparison
    print("\nRunning experiments...")
    qp_results, classical_results = compare_agents(
        env,
        n_episodes=n_episodes,
        n_runs=n_runs,
        verbose=True
    )

    # Print summary
    print_comparison_summary(qp_results, classical_results)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Learning curves
    plot_learning_curves(
        qp_results,
        classical_results,
        window=50,
        save_path=os.path.join(env_output_dir, "learning_curves.png")
    )

    # 2. Comparison bars
    plot_comparison_bars(
        qp_results,
        classical_results,
        save_path=os.path.join(env_output_dir, "comparison_bars.png")
    )

    # 3. Train final agents for visualization
    print("\nTraining final agents for policy visualization...")

    # QP agent
    qp_agent = QuasiProbabilityAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        beta=0.05,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        backward_epsilon=0.1,
        backward_prob=0.3,
        use_backward=True
    )

    from qp_rl_v2.experiment import train_agent
    train_agent(env, qp_agent, n_episodes=n_episodes, eval_every=n_episodes, verbose=False)

    # Classical agent
    classical_agent = ClassicalQAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    train_agent(env, classical_agent, n_episodes=n_episodes, eval_every=n_episodes, verbose=False)

    # 4. Visualize policies
    visualize_policy(
        qp_agent,
        env,
        title="QP-RL Learned Policy",
        save_path=os.path.join(env_output_dir, "policy_qp.png")
    )

    visualize_policy(
        classical_agent,
        env,
        title="Classical Q-Learning Policy",
        save_path=os.path.join(env_output_dir, "policy_classical.png")
    )

    # 5. Visualize quasi-probabilities
    visualize_quasi_probabilities(
        qp_agent,
        env,
        save_path=os.path.join(env_output_dir, "quasi_probabilities.png")
    )

    print(f"\n✓ Experiment complete! Results saved to: {env_output_dir}/")

    # Save summary statistics
    with open(os.path.join(env_output_dir, "summary.txt"), 'w') as f:
        f.write(f"Quasi-Probability RL Experiment Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Environment: {env_display_name}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Runs: {n_runs}\n\n")

        f.write("QP-RL Results:\n")
        qp_rewards = [r.avg_reward for r in qp_results]
        qp_steps = [r.avg_steps for r in qp_results]
        f.write(f"  Avg Reward: {np.mean(qp_rewards):.3f} ± {np.std(qp_rewards):.3f}\n")
        f.write(f"  Avg Steps: {np.mean(qp_steps):.1f} ± {np.std(qp_steps):.1f}\n\n")

        f.write("Classical Q-Learning Results:\n")
        classical_rewards = [r.avg_reward for r in classical_results]
        classical_steps = [r.avg_steps for r in classical_results]
        f.write(f"  Avg Reward: {np.mean(classical_rewards):.3f} ± {np.std(classical_rewards):.3f}\n")
        f.write(f"  Avg Steps: {np.mean(classical_steps):.1f} ± {np.std(classical_steps):.1f}\n\n")

        reward_improvement = (np.mean(qp_rewards) - np.mean(classical_rewards)) / abs(np.mean(classical_rewards)) * 100
        f.write(f"Improvement: {reward_improvement:+.1f}%\n")

    print(f"✓ Summary saved to: {env_output_dir}/summary.txt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Quasi-Probability RL experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on trap maze (default)
  python run_experiments.py

  # Run on long corridor with more episodes
  python run_experiments.py --env long_corridor --episodes 2000

  # Run quick test
  python run_experiments.py --episodes 500 --runs 3

  # Run full suite on all environments
  python run_experiments.py --all --episodes 1500 --runs 5
        """
    )

    parser.add_argument(
        '--env',
        type=str,
        default='trap_maze',
        choices=['trap_maze', 'long_corridor', 'four_rooms'],
        help='Environment to test on'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes per run'
    )

    parser.add_argument(
        '--runs',
        type=int,
        default=5,
        help='Number of independent runs'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results_v2',
        help='Output directory for results'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run experiments on all environments'
    )

    args = parser.parse_args()

    if args.all:
        # Run on all environments
        environments = ['trap_maze', 'long_corridor', 'four_rooms']
        for env_name in environments:
            run_full_experiment(
                env_name=env_name,
                n_episodes=args.episodes,
                n_runs=args.runs,
                output_dir=args.output
            )
            print("\n" + "="*70 + "\n")
    else:
        # Run on single environment
        run_full_experiment(
            env_name=args.env,
            n_episodes=args.episodes,
            n_runs=args.runs,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
