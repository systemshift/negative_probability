#!/usr/bin/env python3
"""
Test counterfactual reasoning approach.

Compare three agents:
1. Classical Q-learning (baseline)
2. QP with backward jumps (original approach)
3. Counterfactual reasoning (no jumps, pure inference)
"""

import numpy as np
from qp_rl_v2.grid_environment import create_trap_maze
from qp_rl_v2.quasi_probability_agent import ClassicalQAgent
from qp_rl_v2.counterfactual_agent import CounterfactualQPAgent, ImprovedCounterfactualAgent
from qp_rl_v2.experiment import run_episode


def train_and_evaluate(env, agent, n_episodes=500, eval_interval=50):
    """Train agent and collect performance metrics."""
    rewards = []
    steps_list = []
    success_count = 0

    for episode in range(n_episodes):
        # Run episode manually (since CounterfactualAgent doesn't work with run_episode)
        state = env.reset()
        agent.reset_episode()

        episode_reward = 0.0
        episode_steps = 0

        for _ in range(env.max_steps):
            action_result = agent.choose_action(state)

            # Handle both tuple (action_type, action) and int (action)
            if isinstance(action_result, tuple):
                action = int(action_result[1])
            else:
                action = int(action_result)

            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        rewards.append(episode_reward)
        steps_list.append(episode_steps)
        if env.agent_pos == env.goal_pos:
            success_count += 1

        agent.decay_epsilon()

        if (episode + 1) % eval_interval == 0:
            recent_rewards = rewards[-eval_interval:]
            recent_success = sum(1 for i in range(max(0, len(rewards)-eval_interval), len(rewards))
                                 if i < len(rewards) and rewards[i] > 0) / eval_interval
            print(f"  Episode {episode+1}: Avg Reward={np.mean(recent_rewards):.2f}, "
                  f"Success Rate={recent_success:.1%}, Epsilon={agent.epsilon:.3f}")

    return rewards, steps_list, success_count


def main():
    print("="*70)
    print("COUNTERFACTUAL REASONING TEST")
    print("="*70)

    env = create_trap_maze()
    print("\nEnvironment: Trap Maze")
    env.render()

    n_episodes = 500
    n_runs = 3

    # Collect results across multiple runs
    classical_all = []
    counterfactual_all = []
    improved_all = []

    for run in range(n_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run+1}/{n_runs}")
        print(f"{'='*70}")

        # 1. Classical Q-Learning
        print("\n[1/3] Training Classical Q-Learning...")
        classical = ClassicalQAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=0.1,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995
        )
        classical_rewards, classical_steps, classical_success = train_and_evaluate(
            env, classical, n_episodes, eval_interval=100
        )
        classical_all.append((classical_rewards, classical_steps, classical_success))

        # 2. Counterfactual Agent (basic)
        print("\n[2/3] Training Counterfactual QP Agent...")
        counterfactual = CounterfactualQPAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=0.1,
            beta=0.05,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            lambda_trace=0.9,
            use_counterfactual=True
        )
        cf_rewards, cf_steps, cf_success = train_and_evaluate(
            env, counterfactual, n_episodes, eval_interval=100
        )
        counterfactual_all.append((cf_rewards, cf_steps, cf_success))

        # 3. Improved Counterfactual Agent
        print("\n[3/3] Training Improved Counterfactual Agent...")
        improved = ImprovedCounterfactualAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=0.1,
            beta=0.05,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            lambda_trace=0.9,
            use_counterfactual=True
        )
        imp_rewards, imp_steps, imp_success = train_and_evaluate(
            env, improved, n_episodes, eval_interval=100
        )
        improved_all.append((imp_rewards, imp_steps, imp_success))

    # Aggregate results
    print("\n" + "="*70)
    print("FINAL RESULTS (averaged over {} runs)".format(n_runs))
    print("="*70)

    def print_stats(name, results_all):
        all_rewards = [r[0] for r in results_all]
        all_steps = [r[1] for r in results_all]
        all_success = [r[2] for r in results_all]

        # Last 100 episodes stats
        final_rewards = [np.mean(r[-100:]) for r in all_rewards]
        final_steps = [np.mean(s[-100:]) for s in all_steps]

        print(f"\n{name}:")
        print(f"  Final Reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
        print(f"  Final Steps: {np.mean(final_steps):.1f} ± {np.std(final_steps):.1f}")
        print(f"  Total Success: {np.mean(all_success):.0f} / {n_episodes}")
        print(f"  Success Rate: {np.mean(all_success)/n_episodes:.1%}")

    print_stats("Classical Q-Learning", classical_all)
    print_stats("Counterfactual QP (Basic)", counterfactual_all)
    print_stats("Counterfactual QP (Improved)", improved_all)

    # Comparison
    print("\n" + "="*70)
    print("IMPROVEMENT vs CLASSICAL")
    print("="*70)

    classical_final = np.mean([np.mean(r[0][-100:]) for r in classical_all])

    cf_final = np.mean([np.mean(r[0][-100:]) for r in counterfactual_all])
    cf_improvement = (cf_final - classical_final) / abs(classical_final) * 100

    imp_final = np.mean([np.mean(r[0][-100:]) for r in improved_all])
    imp_improvement = (imp_final - classical_final) / abs(classical_final) * 100

    print(f"\nCounterfactual (Basic): {cf_improvement:+.1f}%")
    print(f"Counterfactual (Improved): {imp_improvement:+.1f}%")

    if cf_improvement > 0 or imp_improvement > 0:
        print("\n✓ Counterfactual reasoning shows improvement!")
    else:
        print("\n⚠ Counterfactual reasoning needs tuning")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
