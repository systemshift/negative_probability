"""
Experiment Runner for Quasi-Probability RL

Compares QP-RL with classical Q-learning on various environments.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import time
from dataclasses import dataclass, field

from .quasi_probability_agent import QuasiProbabilityAgent, ClassicalQAgent
from .grid_environment import GridWorld


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    agent_name: str
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    success_rate: float = 0.0
    avg_reward: float = 0.0
    avg_steps: float = 0.0
    convergence_episode: int = -1  # Episode where agent "converged"
    training_time: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)


def run_episode(env: GridWorld, agent: Any, training: bool = True) -> Tuple[float, int, bool]:
    """
    Run a single episode.

    Args:
        env: Environment to run in
        agent: Agent to use
        training: Whether to update the agent

    Returns:
        total_reward: Total reward received
        steps: Number of steps taken
        success: Whether goal was reached
    """
    state = env.reset()
    agent.reset_episode()

    total_reward = 0.0
    steps = 0
    success = False

    while True:
        # Choose action
        action_type, action_data = agent.choose_action(state)

        if action_type == "backward" and isinstance(agent, QuasiProbabilityAgent):
            # Perform backward jump
            jump_idx = action_data

            if jump_idx < len(agent.trajectory):
                # Jump to past state
                jump_state, jump_action, _ = agent.trajectory[jump_idx]
                env.set_state(jump_state)
                state = jump_state

                # Truncate trajectory
                agent.trajectory = agent.trajectory[:jump_idx + 1]

            continue  # Don't count this as a step

        # Forward action
        action = action_data
        next_state, reward, done, info = env.step(action)

        # Update agent
        if training:
            agent.update(state, action, reward, next_state, done)

        total_reward += reward
        steps += 1
        state = next_state

        if done:
            if env.agent_pos == env.goal_pos:
                success = True
            break

    return total_reward, steps, success


def train_agent(
    env: GridWorld,
    agent: Any,
    n_episodes: int = 1000,
    eval_every: int = 100,
    verbose: bool = True
) -> ExperimentResults:
    """
    Train an agent and collect results.

    Args:
        env: Environment to train in
        agent: Agent to train
        n_episodes: Number of training episodes
        eval_every: Evaluate every N episodes
        verbose: Print progress

    Returns:
        ExperimentResults object with training results
    """
    results = ExperimentResults(agent_name=agent.__class__.__name__)

    start_time = time.time()
    success_window = []
    converged = False

    for episode in range(n_episodes):
        # Training episode
        reward, steps, success = run_episode(env, agent, training=True)

        results.episode_rewards.append(reward)
        results.episode_steps.append(steps)
        success_window.append(1.0 if success else 0.0)

        # Keep window of last 100 episodes
        if len(success_window) > 100:
            success_window.pop(0)

        # Check for convergence (>80% success in last 100 episodes)
        if not converged and len(success_window) == 100:
            if np.mean(success_window) > 0.8:
                results.convergence_episode = episode
                converged = True

        # Decay exploration
        agent.decay_epsilon()

        # Periodic evaluation and logging
        if verbose and (episode + 1) % eval_every == 0:
            recent_rewards = results.episode_rewards[-eval_every:]
            recent_success = success_window[-min(eval_every, len(success_window)):]

            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {np.mean(recent_rewards):.3f}")
            print(f"  Success Rate: {np.mean(recent_success):.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")

            if isinstance(agent, QuasiProbabilityAgent):
                stats = agent.get_stats()
                if stats['forward_updates'] > 0:
                    jump_rate = stats['backward_jumps'] / episode
                    print(f"  Backward Jumps/Episode: {jump_rate:.2f}")

    results.training_time = time.time() - start_time

    # Final statistics
    results.success_rate = np.mean([1.0 if env.goal_pos == env.goal_pos else 0.0
                                    for _ in range(100)])  # Placeholder
    results.avg_reward = np.mean(results.episode_rewards[-100:])
    results.avg_steps = np.mean(results.episode_steps[-100:])
    results.stats = agent.get_stats()

    return results


def compare_agents(
    env: GridWorld,
    n_episodes: int = 1000,
    n_runs: int = 5,
    verbose: bool = True
) -> Tuple[List[ExperimentResults], List[ExperimentResults]]:
    """
    Compare QP-RL agent with classical Q-learning across multiple runs.

    Args:
        env: Environment to test on
        n_episodes: Number of episodes per run
        n_runs: Number of independent runs
        verbose: Print progress

    Returns:
        qp_results: List of results for QP-RL agent
        classical_results: List of results for classical agent
    """
    qp_results = []
    classical_results = []

    for run in range(n_runs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN {run + 1}/{n_runs}")
            print(f"{'='*60}")

        # Train QP-RL agent
        if verbose:
            print("\n--- Training Quasi-Probability Agent ---")

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

        qp_result = train_agent(env, qp_agent, n_episodes, eval_every=n_episodes//10, verbose=verbose)
        qp_results.append(qp_result)

        # Train classical agent
        if verbose:
            print("\n--- Training Classical Q-Learning Agent ---")

        classical_agent = ClassicalQAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=0.1,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995
        )

        classical_result = train_agent(env, classical_agent, n_episodes, eval_every=n_episodes//10, verbose=verbose)
        classical_results.append(classical_result)

    return qp_results, classical_results


def print_comparison_summary(
    qp_results: List[ExperimentResults],
    classical_results: List[ExperimentResults]
):
    """Print summary comparison of results."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    # Average across runs
    qp_rewards = [r.avg_reward for r in qp_results]
    classical_rewards = [r.avg_reward for r in classical_results]

    qp_steps = [r.avg_steps for r in qp_results]
    classical_steps = [r.avg_steps for r in classical_results]

    qp_convergence = [r.convergence_episode for r in qp_results if r.convergence_episode != -1]
    classical_convergence = [r.convergence_episode for r in classical_results if r.convergence_episode != -1]

    print("\nQuasi-Probability Agent:")
    print(f"  Avg Reward: {np.mean(qp_rewards):.3f} ± {np.std(qp_rewards):.3f}")
    print(f"  Avg Steps:  {np.mean(qp_steps):.1f} ± {np.std(qp_steps):.1f}")
    if qp_convergence:
        print(f"  Convergence Episode: {np.mean(qp_convergence):.0f} ± {np.std(qp_convergence):.0f}")
    else:
        print(f"  Convergence Episode: Did not converge")

    print("\nClassical Q-Learning Agent:")
    print(f"  Avg Reward: {np.mean(classical_rewards):.3f} ± {np.std(classical_rewards):.3f}")
    print(f"  Avg Steps:  {np.mean(classical_steps):.1f} ± {np.std(classical_steps):.1f}")
    if classical_convergence:
        print(f"  Convergence Episode: {np.mean(classical_convergence):.0f} ± {np.std(classical_convergence):.0f}")
    else:
        print(f"  Convergence Episode: Did not converge")

    # Statistical comparison
    print("\nComparison:")
    reward_improvement = (np.mean(qp_rewards) - np.mean(classical_rewards)) / abs(np.mean(classical_rewards)) * 100
    steps_improvement = (np.mean(classical_steps) - np.mean(qp_steps)) / np.mean(classical_steps) * 100

    print(f"  Reward Improvement: {reward_improvement:+.1f}%")
    print(f"  Steps Improvement:  {steps_improvement:+.1f}%")

    if qp_convergence and classical_convergence:
        convergence_improvement = (np.mean(classical_convergence) - np.mean(qp_convergence)) / np.mean(classical_convergence) * 100
        print(f"  Faster Convergence: {convergence_improvement:+.1f}%")

    # Backward jump statistics
    if qp_results[0].stats.get('backward_jumps', 0) > 0:
        avg_jumps = np.mean([r.stats['backward_jumps'] for r in qp_results])
        print(f"\n  Avg Backward Jumps: {avg_jumps:.0f}")

    print("="*60)


if __name__ == "__main__":
    from .grid_environment import create_trap_maze, create_long_corridor

    print("Testing Experiment Runner\n")

    # Quick test on trap maze
    env = create_trap_maze()
    print("Environment: Trap Maze")
    env.render()

    print("\nRunning comparison (1 run, 500 episodes each)...")
    qp_results, classical_results = compare_agents(
        env,
        n_episodes=500,
        n_runs=1,
        verbose=True
    )

    print_comparison_summary(qp_results, classical_results)
