#!/usr/bin/env python3
"""
Comprehensive comparison of Standard MCTS vs QP-MCTS.

Tests on tic-tac-toe to validate that quasi-probabilities help.
"""

import numpy as np
from tictactoe import TicTacToe
from mcts import MCTS, QPMCTS


def play_game(mcts1, mcts2, verbose=False):
    """
    Play one game between two MCTS agents.

    Args:
        mcts1: MCTS agent for player 1
        mcts2: MCTS agent for player -1
        verbose: Print game progress

    Returns:
        winner: 1, -1, or 0 for draw
    """
    game = TicTacToe()

    if verbose:
        print("\nStarting game...")
        game.render()

    while not game.is_terminal():
        if game.current_player == 1:
            # Player 1's turn
            root = mcts1.search(game)
            action = mcts1.get_best_action()
        else:
            # Player -1's turn
            root = mcts2.search(game)
            action = mcts2.get_best_action()

        if verbose:
            player_name = "X" if game.current_player == 1 else "O"
            print(f"{player_name} plays {action}")

        game = game.make_move(action)

        if verbose:
            game.render()

    winner = game.get_winner()

    if verbose:
        if winner == 1:
            print("X wins!")
        elif winner == -1:
            print("O wins!")
        else:
            print("Draw!")

    return winner


def tournament(agent1_class, agent2_class, num_games=50, simulations=100):
    """
    Run tournament between two agent types.

    Args:
        agent1_class: MCTS or QPMCTS class
        agent2_class: MCTS or QPMCTS class
        num_games: Number of games to play
        simulations: Simulations per move

    Returns:
        Dictionary with results
    """
    print(f"\nTournament: {agent1_class.__name__} vs {agent2_class.__name__}")
    print(f"Games: {num_games}, Simulations: {simulations}")
    print("-" * 60)

    wins_1 = 0
    wins_2 = 0
    draws = 0

    nodes_1 = []
    nodes_2 = []

    for game_num in range(num_games):
        # Alternate who goes first
        if game_num % 2 == 0:
            agent1 = agent1_class(num_simulations=simulations)
            agent2 = agent2_class(num_simulations=simulations)
            player1_is_x = True
        else:
            agent2 = agent2_class(num_simulations=simulations)
            agent1 = agent1_class(num_simulations=simulations)
            player1_is_x = False

        winner = play_game(agent1, agent2, verbose=False)

        # Track nodes explored
        nodes_1.append(agent1.nodes_explored)
        nodes_2.append(agent2.nodes_explored)

        # Record outcome from perspective of agent1
        if player1_is_x:
            if winner == 1:
                wins_1 += 1
            elif winner == -1:
                wins_2 += 1
            else:
                draws += 1
        else:
            if winner == -1:
                wins_1 += 1
            elif winner == 1:
                wins_2 += 1
            else:
                draws += 1

        if (game_num + 1) % 10 == 0:
            print(f"Games {game_num + 1}/{num_games}: "
                  f"{agent1_class.__name__} {wins_1}W {draws}D {wins_2}L")

    print("\nFinal Results:")
    print(f"  {agent1_class.__name__}: {wins_1} wins ({wins_1/num_games:.1%})")
    print(f"  {agent2_class.__name__}: {wins_2} wins ({wins_2/num_games:.1%})")
    print(f"  Draws: {draws} ({draws/num_games:.1%})")

    print(f"\nNodes Explored (average per game):")
    print(f"  {agent1_class.__name__}: {np.mean(nodes_1):.0f} ± {np.std(nodes_1):.0f}")
    print(f"  {agent2_class.__name__}: {np.mean(nodes_2):.0f} ± {np.std(nodes_2):.0f}")

    return {
        'agent1': agent1_class.__name__,
        'agent2': agent2_class.__name__,
        'wins_1': wins_1,
        'wins_2': wins_2,
        'draws': draws,
        'nodes_1': nodes_1,
        'nodes_2': nodes_2
    }


def test_node_efficiency():
    """
    Test: How many simulations needed to find optimal first move?

    Optimal first move in tic-tac-toe is center (1,1).
    """
    print("\n" + "=" * 60)
    print("NODE EFFICIENCY TEST")
    print("=" * 60)
    print("Question: How many simulations to find optimal first move?")
    print("(Optimal = center (1,1))")

    game = TicTacToe()
    simulation_counts = [10, 25, 50, 100, 200, 500]

    print("\nStandard MCTS:")
    for sims in simulation_counts:
        mcts = MCTS(num_simulations=sims)
        root = mcts.search(game)
        best_action = mcts.get_best_action()
        is_optimal = (best_action == (1, 1))
        print(f"  {sims:4d} sims: {best_action} {'✓ Optimal' if is_optimal else '✗ Suboptimal'}")

    print("\nQP-MCTS:")
    for sims in simulation_counts:
        qp_mcts = QPMCTS(num_simulations=sims, alpha=0.1)
        root = qp_mcts.search(game)
        best_action = qp_mcts.get_best_action()
        is_optimal = (best_action == (1, 1))
        print(f"  {sims:4d} sims: {best_action} {'✓ Optimal' if is_optimal else '✗ Suboptimal'}")


def test_from_position():
    """
    Test: Performance from specific position.

    X . O
    . X .
    . . .

    X to move - should block at (0,1) or win at (2,2)
    """
    print("\n" + "=" * 60)
    print("TACTICAL POSITION TEST")
    print("=" * 60)

    game = TicTacToe()
    game = game.make_move((0, 0))  # X
    game = game.make_move((0, 2))  # O
    game = game.make_move((1, 1))  # X

    print("Position (X to move):")
    game.render()
    print("X can win immediately with (2,2)")

    print("\nStandard MCTS:")
    for sims in [10, 25, 50, 100]:
        mcts = MCTS(num_simulations=sims)
        root = mcts.search(game)
        best_action = mcts.get_best_action()
        found_win = (best_action == (2, 2))
        print(f"  {sims:3d} sims: {best_action} {'✓ Found win' if found_win else '✗ Missed'}")

    print("\nQP-MCTS:")
    for sims in [10, 25, 50, 100]:
        qp_mcts = QPMCTS(num_simulations=sims, alpha=0.1)
        root = qp_mcts.search(game)
        best_action = qp_mcts.get_best_action()
        found_win = (best_action == (2, 2))
        print(f"  {sims:3d} sims: {best_action} {'✓ Found win' if found_win else '✗ Missed'}")


def main():
    print("=" * 60)
    print("QP-MCTS vs Standard MCTS: Comprehensive Comparison")
    print("=" * 60)

    # Test 1: Node efficiency
    test_node_efficiency()

    # Test 2: Tactical position
    test_from_position()

    # Test 3: Self-play tournament
    print("\n" + "=" * 60)
    print("SELF-PLAY TOURNAMENTS")
    print("=" * 60)

    # QP vs QP (sanity check)
    tournament(QPMCTS, QPMCTS, num_games=20, simulations=100)

    # Standard vs Standard (sanity check)
    tournament(MCTS, MCTS, num_games=20, simulations=100)

    # Main comparison: QP-MCTS vs Standard MCTS
    print("\n" + "=" * 60)
    print("MAIN COMPARISON")
    print("=" * 60)
    results = tournament(QPMCTS, MCTS, num_games=50, simulations=100)

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    qp_wins = results['wins_1']
    std_wins = results['wins_2']
    draws = results['draws']
    total = qp_wins + std_wins + draws

    print(f"\nWin Rate:")
    print(f"  QP-MCTS: {qp_wins/total:.1%}")
    print(f"  Standard: {std_wins/total:.1%}")
    print(f"  Draw: {draws/total:.1%}")

    qp_nodes = np.mean(results['nodes_1'])
    std_nodes = np.mean(results['nodes_2'])

    print(f"\nComputational Efficiency:")
    print(f"  Nodes per game (QP): {qp_nodes:.0f}")
    print(f"  Nodes per game (Std): {std_nodes:.0f}")
    print(f"  Difference: {(qp_nodes - std_nodes)/std_nodes * 100:+.1f}%")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if qp_wins > std_wins * 1.2:
        print("✓ QP-MCTS shows clear advantage!")
        print("  Recommendation: Test on harder game (chess)")
    elif qp_wins > std_wins:
        print("✓ QP-MCTS shows slight advantage")
        print("  Recommendation: Tune parameters, test more games")
    elif abs(qp_wins - std_wins) <= total * 0.1:
        print("≈ Both methods perform similarly")
        print("  Recommendation: Check if quasi-probs helping with efficiency")
    else:
        print("✗ Standard MCTS performing better")
        print("  Recommendation: Debug quasi-probability updates")

    print("\nKey insight: Check quasi-probability statistics to see if")
    print("negative values correctly identify bad moves.")


if __name__ == "__main__":
    main()
