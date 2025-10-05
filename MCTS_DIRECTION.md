# Quasi-Probability Monte Carlo Tree Search: A Promising Research Direction

## Executive Summary

Your quasi-probability RL project has been struggling with model-free grid worlds because you lack knowledge of environment dynamics. However, your core insight about **backward reasoning and counterfactual thinking** aligns perfectly with **Monte Carlo Tree Search (MCTS)** - the algorithm behind AlphaZero's success in chess and Go.

**Key Insight**: MCTS already does backward propagation of information through its backpropagation phase. Quasi-probabilities could enhance this by explicitly representing "refuted paths" with negative values.

**Recommendation**: Pivot from model-free RL in grid worlds → Tree search and planning algorithms where you have (or can learn) perfect dynamics.

---

## Table of Contents

1. [How AlphaZero Actually Works](#how-alphazero-actually-works)
2. [Connection to Quasi-Probabilities](#connection-to-quasi-probabilities)
3. [Why MCTS Works Where Model-Free RL Struggles](#why-mcts-works-where-model-free-rl-struggles)
4. [Research Direction: Quasi-Probability MCTS](#research-direction-quasi-probability-mcts)
5. [Concrete Implementation Plan](#concrete-implementation-plan)
6. [Literature Review](#literature-review)
7. [Experimental Protocol](#experimental-protocol)
8. [Publication Strategy](#publication-strategy)
9. [Code Architecture](#code-architecture)
10. [Timeline and Milestones](#timeline-and-milestones)

---

## How AlphaZero Actually Works

### The Architecture

AlphaZero combines two components that work synergistically:

#### 1. Deep Neural Network

The network has two heads:

**Policy Head** π(a|s):
- Input: Board position (encoded as tensor)
- Output: Probability distribution over legal moves
- Purpose: "What moves look promising?"

**Value Head** V(s):
- Input: Same board position
- Output: Scalar value ∈ [-1, 1]
- Purpose: "Who's winning in this position?"

```python
class AlphaZeroNetwork(nn.Module):
    def __init__(self):
        self.shared_layers = ResNet(...)
        self.policy_head = nn.Linear(...)  # → move probabilities
        self.value_head = nn.Linear(...)    # → position evaluation

    def forward(self, position):
        features = self.shared_layers(position)
        policy = softmax(self.policy_head(features))
        value = tanh(self.value_head(features))
        return policy, value
```

#### 2. Monte Carlo Tree Search (MCTS)

MCTS builds a tree of game positions through four phases:

**Phase 1: Selection**
```
Start at root
While not at leaf:
    Pick child with highest UCB score:
    UCB(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Q(s,a) = average value from this action
    P(s,a) = neural network prior probability
    N(s,a) = visit count for this action
```

**Phase 2: Expansion**
```
At leaf node:
    Get policy, value from neural network
    Add children for all legal moves
```

**Phase 3: Evaluation**
```
Use neural network to evaluate leaf position:
    V(s_leaf) ← network.value_head(s_leaf)
```

**Phase 4: Backpropagation** ← **This is the key!**
```
Walk back up tree from leaf to root:
    For each edge (s,a) on path:
        N(s,a) += 1
        Q(s,a) = (Q(s,a) * (N-1) + V) / N

Information flows BACKWARD through tree!
```

### The Training Loop

AlphaZero learns through **self-play**:

```python
def train_alphazero():
    network = initialize_network()

    while not converged:
        # 1. Generate training data
        games = []
        for _ in range(num_games):
            game = []
            state = initial_position()

            while not terminal(state):
                # Use MCTS with current network
                mcts = MCTS(network)
                for _ in range(num_simulations):
                    mcts.search(state)

                # Get move distribution from MCTS
                pi = mcts.get_action_probs(state)
                game.append((state, pi))

                # Make move
                action = sample(pi)
                state = make_move(state, action)

            # Game over - record outcome
            z = game_outcome()  # +1 win, -1 loss, 0 draw
            for (s, pi) in game:
                games.append((s, pi, z))

        # 2. Train network on self-play data
        for (state, target_policy, target_value) in games:
            pred_policy, pred_value = network(state)

            loss = cross_entropy(pred_policy, target_policy) + \
                   mse(pred_value, target_value)

            optimizer.step(loss)

        # 3. Repeat with improved network
```

**Key insight**: MCTS uses network to guide search, network learns from MCTS's improved play. This creates a virtuous cycle.

---

## Connection to Quasi-Probabilities

### MCTS Already Does "Backward" Reasoning!

The **backpropagation phase** of MCTS propagates information backward:

```
Final position: Win (+1)
    ↓ backprop
Move 10: "Ah, this led to win, increase Q-value"
    ↓ backprop
Move 9: "The path through move 10 is good"
    ↓ backprop
Move 8: "This move eventually led to win"
...
    ↓ backprop
Move 1: Value updated based on final outcome
```

This is **exactly** the counterfactual reasoning you wanted!

When the game ends, you learn:
- "What if I had played differently at move 5?"
- MCTS explores alternative moves and updates their Q-values
- Bad moves get negative updates (lower Q)
- Good moves get positive updates (higher Q)

### Quasi-Probabilities in MCTS Context

Current MCTS uses Q-values that can be positive or negative:
- Q(s,a) > 0 means "good for current player"
- Q(s,a) < 0 means "bad for current player"

But these aren't quite "quasi-probabilities" - they're averaged outcomes.

**Your contribution**: Explicitly treat them as quasi-probability distributions:

```python
# Standard MCTS
Q(s,a) = average_value  # Just a number

# Quasi-Probability MCTS
Q_forward(s,a) = prob_this_leads_to_win    # Positive: future potential
Q_backward(s,a) = prob_this_was_mistake     # Negative: past regret
Q_total(s,a) = Q_forward(s,a) - Q_backward(s,a)
```

Interpretation:
- **Q_forward > 0**: This move might lead to victory (explore it)
- **Q_backward > 0**: This move is likely a mistake (avoid it)
- Sum preserves "quasi-probability" constraint

---

## Why MCTS Works Where Model-Free RL Struggles

### Model-Free RL Challenges (Your Current Approach)

| Challenge | Why It's Hard | Impact |
|-----------|---------------|---------|
| **No Dynamics Model** | Don't know P(s'|s,a) | Can't predict counterfactuals accurately |
| **Experience Required** | Must visit state to learn value | Sample inefficient |
| **Credit Assignment** | Hard to propagate value backward | Slow learning |
| **Exploration** | Random exploration wastes samples | Poor in sparse reward |

### MCTS Advantages

| Advantage | Why It Helps | Impact |
|-----------|--------------|---------|
| **Perfect Simulator** | Game rules = perfect model | Can explore any path |
| **Tree Structure** | Explicit representation of explored paths | Clear parent-child relationships |
| **Lookahead** | Think before acting | Much better than greedy |
| **Guided Exploration** | UCB balances exploration/exploitation | Efficient search |
| **Proven Convergence** | Mathematically guaranteed to find optimal play | Reliable |

### Side-by-Side Comparison

**Your QP-RL trying to solve maze**:
```
Problem: Don't know what happens if I go left instead of right
Solution attempted: Guess based on optimism
Result: Guess is often wrong → corrupts learning
```

**MCTS playing chess**:
```
Problem: Don't know if this move is good
Solution: Simulate the move! Play it out!
Result: Actual evaluation from simulation → accurate learning
```

**The key difference**: MCTS can **actually try** the alternative paths because it has a simulator.

---

## Research Direction: Quasi-Probability MCTS

### Core Research Question

**Can negative probabilities improve Monte Carlo Tree Search by explicitly representing refuted/unlikely paths?**

### Hypothesis

Standard MCTS averages outcomes. Our hypothesis: explicitly tracking "negative probability" for bad paths will:
1. **Faster pruning**: Quickly identify and avoid bad branches
2. **Better exploration**: More clearly distinguish promising vs. refuted paths
3. **Transfer learning**: Negative patterns transfer to similar positions

### Proposed Algorithm: QP-MCTS

#### Node Structure

```python
class QPNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action → QPNode

        # Standard MCTS stats
        self.visits = 0
        self.value_sum = 0.0

        # Quasi-probability components (NEW!)
        self.Q_positive = 0.0   # Probability this is good path
        self.Q_negative = 0.0   # Probability this is bad path
        self.Q_quasi = 0.0      # Q_positive - Q_negative

        # For AlphaZero-style
        self.prior_policy = {}  # action → P(a|s) from network
```

#### Selection with Quasi-Probabilities

```python
def select_action(node):
    """
    Select child using modified UCB with quasi-probabilities.
    """
    best_score = -float('inf')
    best_action = None

    for action, child in node.children.items():
        if child.visits == 0:
            # Unvisited - high priority
            score = float('inf')
        else:
            # Standard components
            exploitation = child.Q_quasi  # Using quasi-prob value

            # Exploration bonus (standard UCB)
            exploration = C * sqrt(log(node.visits) / child.visits)

            # Negative probability penalty (NEW!)
            # Avoid paths we've determined are bad
            refutation_penalty = -child.Q_negative

            score = exploitation + exploration + refutation_penalty

        if score > best_score:
            best_score = score
            best_action = action

    return best_action
```

#### Backpropagation with Quasi-Probabilities

```python
def backpropagate(node, value):
    """
    Propagate value up tree, updating quasi-probabilities.
    """
    current = node

    while current is not None:
        current.visits += 1
        current.value_sum += value

        # Update quasi-probability components
        if value > 0:
            # Good outcome - increase positive probability
            alpha = 0.1
            current.Q_positive += alpha * (1.0 - current.Q_positive)
        else:
            # Bad outcome - increase negative probability
            beta = 0.1
            current.Q_negative += beta * (1.0 - current.Q_negative)

        # Quasi-probability is difference
        current.Q_quasi = current.Q_positive - current.Q_negative

        # Normalize to maintain quasi-probability constraint
        # Sum of |Q| over siblings should be reasonable
        if current.parent is not None:
            normalize_siblings(current.parent)

        # Flip value for parent (opponent's turn)
        value = -value
        current = current.parent
```

#### Normalization to Maintain Quasi-Probability Distribution

```python
def normalize_siblings(parent_node):
    """
    Ensure quasi-probabilities of siblings form valid distribution.

    Constraint: Σ|Q_quasi(s,a)| should sum to approximately 1
    """
    children = list(parent_node.children.values())

    if not children:
        return

    # Sum of absolute quasi-probabilities
    total_abs = sum(abs(child.Q_quasi) for child in children)

    if total_abs > 0:
        # Normalize
        for child in children:
            child.Q_quasi /= total_abs
            # Update positive/negative components proportionally
            if child.Q_quasi >= 0:
                child.Q_positive = child.Q_quasi
                child.Q_negative = 0
            else:
                child.Q_positive = 0
                child.Q_negative = -child.Q_quasi
```

### Theoretical Properties

#### Claim 1: Faster Pruning

**Intuition**: Negative probabilities allow us to more confidently prune bad branches.

**Formal**: Let T be the search tree. A branch B is pruned if Q_negative(B) > threshold.

Standard MCTS: Prunes when Q(B) < Q(best) - some margin
QP-MCTS: Also prunes when Q_negative(B) > τ, even if Q(B) is uncertain

**Benefit**: Can rule out paths earlier even with limited visits.

#### Claim 2: Better Exploration

**Intuition**: Quasi-probabilities give richer signal than scalar Q-values.

**Example**:
- Node A: Q=0.1, visited 10 times, all wins → Q_positive=1.0, Q_negative=0
- Node B: Q=0.1, visited 10 times, mixed results → Q_positive=0.5, Q_negative=0.4

Same Q-value, but A is clearly better! QP-MCTS distinguishes them.

#### Claim 3: Transfer Learning

**Intuition**: Negative patterns (known mistakes) transfer across similar positions.

**Example in chess**: If "moving queen early" has Q_negative=0.8 in position P1,
we can initialize Q_negative=0.6 for similar move in position P2.

---

## Concrete Implementation Plan

### Phase 1: Proof of Concept (1-2 weeks)

**Goal**: Verify basic idea works on toy problem.

#### Step 1.1: Implement QP-MCTS for Tic-Tac-Toe

Why tic-tac-toe:
- ✅ Perfect simulator (trivial game rules)
- ✅ Small state space (~5000 positions)
- ✅ Known optimal play
- ✅ Fast to test
- ✅ Can verify correctness

**Code structure**:
```python
# tictactoe_qp_mcts.py

class TicTacToe:
    """Simple tic-tac-toe environment."""
    def __init__(self):
        self.board = np.zeros((3, 3))

    def legal_actions(self):
        return [(i, j) for i in range(3) for j in range(3)
                if self.board[i, j] == 0]

    def make_move(self, action):
        # Returns new state
        pass

    def is_terminal(self):
        # Check win/draw
        pass

    def get_value(self):
        # +1 win, -1 loss, 0 draw
        pass

class QPMCTSAgent:
    """Quasi-probability MCTS agent."""
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        self.root = None

    def search(self, state):
        self.root = QPNode(state)

        for _ in range(self.num_simulations):
            # 1. Selection
            node = self._select(self.root)

            # 2. Expansion
            if not node.is_terminal():
                node = self._expand(node)

            # 3. Simulation
            value = self._simulate(node)

            # 4. Backpropagation
            self._backpropagate(node, value)

    def get_best_action(self):
        # Return action with highest Q_quasi
        pass
```

**Evaluation**:
```python
def test_tictactoe():
    """
    Compare Standard MCTS vs QP-MCTS on tic-tac-toe.
    """
    standard_mcts = StandardMCTS(num_simulations=100)
    qp_mcts = QPMCTSAgent(num_simulations=100)

    # Metrics to track
    metrics = {
        'standard': {'nodes_explored': [], 'win_rate': []},
        'qp': {'nodes_explored': [], 'win_rate': []}
    }

    # Play 100 games
    for game in range(100):
        state = TicTacToe()

        while not state.is_terminal():
            # Standard MCTS move
            standard_mcts.search(state)
            action = standard_mcts.get_best_action()

            # QP-MCTS move
            qp_mcts.search(state)
            action = qp_mcts.get_best_action()

            state.make_move(action)

        # Record metrics
        metrics['standard']['nodes_explored'].append(standard_mcts.total_nodes)
        metrics['qp']['nodes_explored'].append(qp_mcts.total_nodes)

    # Analysis
    print("Standard MCTS: {:.0f} nodes/game".format(
        np.mean(metrics['standard']['nodes_explored'])
    ))
    print("QP-MCTS: {:.0f} nodes/game".format(
        np.mean(metrics['qp']['nodes_explored'])
    ))

    if np.mean(metrics['qp']['nodes_explored']) < \
       np.mean(metrics['standard']['nodes_explored']):
        print("✓ QP-MCTS explores fewer nodes!")
    else:
        print("✗ No improvement in efficiency")
```

**Success criteria**:
- QP-MCTS plays optimally (no losses to random)
- QP-MCTS explores ≤ nodes compared to standard MCTS
- Negative probabilities clearly mark bad moves

#### Step 1.2: Analyze Why It Works (or Doesn't)

**Visualizations**:
```python
def visualize_tree(root_node):
    """
    Plot the MCTS tree colored by quasi-probabilities.

    - Green nodes: High Q_positive (good paths)
    - Red nodes: High Q_negative (refuted paths)
    - Gray nodes: Uncertain
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    colors = []

    def add_node_recursive(node, depth=0):
        node_id = id(node)
        G.add_node(node_id,
                   label=f"Q+:{node.Q_positive:.2f}\nQ-:{node.Q_negative:.2f}",
                   depth=depth)

        # Color based on quasi-probability
        if node.Q_positive > 0.6:
            colors.append('green')
        elif node.Q_negative > 0.6:
            colors.append('red')
        else:
            colors.append('gray')

        for action, child in node.children.items():
            child_id = id(child)
            G.add_edge(node_id, child_id, label=str(action))
            add_node_recursive(child, depth+1)

    add_node_recursive(root_node)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, with_labels=True)
    plt.savefig("mcts_tree_visualization.png")
```

**Key questions**:
1. Do negative probabilities concentrate on bad moves?
2. Does pruning happen earlier than standard MCTS?
3. Is exploration more focused?

### Phase 2: Scale to Chess (2-3 weeks)

**Goal**: Test on real game with large branching factor.

#### Step 2.1: Integrate with Python-Chess Library

```python
import chess

class ChessQPMCTS:
    def __init__(self, num_simulations=800):
        self.num_simulations = num_simulations

    def search(self, board: chess.Board):
        """
        Run QP-MCTS on chess position.
        """
        root = QPNode(board)

        for _ in range(self.num_simulations):
            node = self._select(root)

            if not node.board.is_game_over():
                node = self._expand(node)

            value = self._evaluate(node)
            self._backpropagate(node, value)

        return self._get_best_move(root)

    def _evaluate(self, node):
        """
        Evaluate leaf position.

        Options:
        1. Random playout (weak but fast)
        2. Simple heuristic (material count)
        3. Neural network (strong but slow)
        """
        # Simple heuristic for now
        return self._material_evaluation(node.board)
```

#### Step 2.2: Compare Against Baseline

**Experiment setup**:
```python
def chess_experiment():
    """
    Compare QP-MCTS vs Standard MCTS on chess.
    """
    positions = load_test_positions()  # Famous positions

    results = {
        'standard_mcts': [],
        'qp_mcts': []
    }

    for position in positions:
        # Standard MCTS
        standard = StandardMCTS(num_simulations=800)
        start = time.time()
        move_std = standard.search(position)
        time_std = time.time() - start

        # QP-MCTS
        qp = ChessQPMCTS(num_simulations=800)
        start = time.time()
        move_qp = qp.search(position)
        time_qp = time.time() - start

        results['standard_mcts'].append({
            'move': move_std,
            'time': time_std,
            'nodes': standard.total_nodes,
            'quality': evaluate_move_with_stockfish(move_std)
        })

        results['qp_mcts'].append({
            'move': move_qp,
            'time': time_qp,
            'nodes': qp.total_nodes,
            'quality': evaluate_move_with_stockfish(move_qp)
        })

    # Analysis
    print_comparison_table(results)
```

**Metrics**:
- Nodes explored per search
- Time per move
- Move quality (compare to Stockfish evaluation)
- Pruning rate (% of tree avoided)

#### Step 2.3: Test Against Stockfish

```python
def play_against_stockfish():
    """
    Play full games: QP-MCTS vs Stockfish (weak setting).
    """
    qp_mcts = ChessQPMCTS(num_simulations=1000)

    wins, losses, draws = 0, 0, 0

    for game_num in range(100):
        board = chess.Board()

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # QP-MCTS plays white
                move = qp_mcts.search(board)
            else:
                # Stockfish plays black (limited depth)
                move = get_stockfish_move(board, depth=5)

            board.push(move)

        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1

    print(f"vs Stockfish depth-5: {wins}W {draws}D {losses}L")
```

**Success criteria**:
- QP-MCTS finds good moves faster than standard MCTS
- Comparable quality with fewer simulations
- Clear improvement on tactics puzzles (forced wins)

### Phase 3: Add Neural Network (3-4 weeks)

**Goal**: Combine with AlphaZero-style learning.

#### Step 3.1: Design Quasi-Probability Network

```python
class QPNetwork(nn.Module):
    """
    Neural network with quasi-probability outputs.
    """
    def __init__(self):
        super().__init__()

        # Shared representation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(17, 256, 3, padding=1),  # Chess: 17 planes
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            # ... more layers
        )

        # Policy head (standard)
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, 1),
            nn.Flatten(),
            nn.Linear(2*8*8, 1968)  # Chess: 1968 possible moves
        )

        # Value head (standard)
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Flatten(),
            nn.Linear(1*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        # Quasi-probability head (NEW!)
        self.qp_positive_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Flatten(),
            nn.Linear(1*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # [0, 1]
        )

        self.qp_negative_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Flatten(),
            nn.Linear(1*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # [0, 1]
        )

    def forward(self, board_tensor):
        features = self.conv_layers(board_tensor)

        policy = F.softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features)
        qp_positive = self.qp_positive_head(features)
        qp_negative = self.qp_negative_head(features)

        return policy, value, qp_positive, qp_negative
```

#### Step 3.2: Training Loop

```python
def train_qp_alphazero():
    """
    Train QP-Network using self-play.
    """
    network = QPNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    for iteration in range(num_iterations):
        # 1. Self-play
        games = []
        for _ in range(num_games):
            game_data = self_play_with_qp_mcts(network)
            games.extend(game_data)

        # 2. Train network
        for batch in batches(games):
            positions, target_pi, target_v, target_qp = batch

            pred_pi, pred_v, pred_qp_pos, pred_qp_neg = network(positions)

            loss = (
                cross_entropy(pred_pi, target_pi) +
                mse(pred_v, target_v) +
                mse(pred_qp_pos, target_qp['positive']) +
                mse(pred_qp_neg, target_qp['negative'])
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate
        if iteration % 10 == 0:
            evaluate_network(network)
```

#### Step 3.3: Self-Play with QP-MCTS

```python
def self_play_with_qp_mcts(network):
    """
    Play one game using QP-MCTS guided by network.
    """
    game_data = []
    board = chess.Board()

    while not board.is_game_over():
        # Run QP-MCTS
        mcts = QPMCTS(network)
        for _ in range(800):
            mcts.search(board)

        # Get move probabilities from MCTS
        pi = mcts.get_action_probs(board)

        # Get quasi-probability components from tree
        qp_positive, qp_negative = mcts.get_quasi_probs(board)

        # Store training data
        game_data.append((
            encode_board(board),
            pi,
            qp_positive,
            qp_negative
        ))

        # Make move
        move = sample(pi)
        board.push(move)

    # Game over - label all positions with final result
    z = get_game_outcome(board)

    return [(pos, pi, z, qp_pos, qp_neg)
            for (pos, pi, qp_pos, qp_neg) in game_data]
```

**Success criteria**:
- Network learns to predict quasi-probabilities accurately
- QP-guided MCTS outperforms standard AlphaZero
- Training converges faster

### Phase 4: Evaluation & Publication (2-3 weeks)

**Goal**: Rigorous evaluation and paper writing.

#### Evaluation Protocol

**Benchmark 1: Node Efficiency**
```
Test: How many nodes to find best move?
Positions: 100 tactical puzzles
Metric: Nodes explored until optimal move found
Expected: QP-MCTS < Standard MCTS
```

**Benchmark 2: Time to Solution**
```
Test: How fast to solve forced mate problems?
Positions: Mate-in-3, Mate-in-4, Mate-in-5
Metric: Time until mate sequence found
Expected: QP-MCTS faster due to better pruning
```

**Benchmark 3: Playing Strength**
```
Test: Win rate against strong opponents
Opponents: Stockfish (various depths), standard AlphaZero
Metric: Elo rating, win/draw/loss
Expected: Comparable strength, better sample efficiency
```

**Benchmark 4: Pruning Analysis**
```
Test: How much of tree is pruned?
Analysis: Track Q_negative values, pruning decisions
Metric: % of branches avoided, correctness of pruning
Expected: More aggressive pruning without sacrificing quality
```

---

## Literature Review

### Core Papers to Read

#### MCTS Fundamentals

1. **"Bandit Based Monte-Carlo Planning"** (Kocsis & Szepesvári, 2006)
   - Original UCT algorithm
   - Theoretical foundations
   - Connection to multi-armed bandits

2. **"A Survey of Monte Carlo Tree Search Methods"** (Browne et al., 2012)
   - Comprehensive review
   - Variants and applications
   - Best practices

#### AlphaGo/AlphaZero

3. **"Mastering the Game of Go with Deep Neural Networks"** (Silver et al., 2016)
   - AlphaGo architecture
   - Policy and value networks
   - Training methodology

4. **"Mastering Chess and Shogi by Self-Play with a General RL Algorithm"** (Silver et al., 2017)
   - AlphaZero: generalization to chess
   - Pure self-play learning
   - No human knowledge

5. **"Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"** (Schrittwieser et al., 2019)
   - MuZero: learns dynamics model
   - MCTS with learned model
   - State-of-the-art results

#### Relevant RL Theory

6. **"Hindsight Experience Replay"** (Andrychowicz et al., 2017)
   - Learn from failed trajectories
   - Relabel goals retrospectively
   - Most similar to your counterfactual idea

7. **"World Models"** (Ha & Schmidhuber, 2018)
   - Learn environment dynamics
   - Use model for planning
   - Relevant for model-based extension

#### Negative Probabilities in Physics

8. **"Negative Probability"** (Feynman, 1987)
   - Philosophical foundations
   - Quasi-probability distributions
   - Quantum mechanics interpretation

9. **"The Wigner Function: I. The Physical Interpretation"** (Hillery et al., 1984)
   - Quasi-probability in quantum mechanics
   - Negative regions have physical meaning
   - Mathematical framework

#### Tree Search Improvements

10. **"Monte Carlo Tree Search with Heuristic Evaluations"** (Huang et al., 2011)
    - Combining MCTS with heuristics
    - Evaluation functions
    - Pruning strategies

11. **"Best-First Minimax Search"** (Korf & Chickering, 1996)
    - Alternative to MCTS
    - Proof-number search
    - Relevant for comparison

### Gap in Literature

**What's missing**:
- No formal framework for "anti-values" in tree search
- No explicit representation of refuted paths
- Limited work on quasi-probabilities outside quantum mechanics
- No connection between negative probabilities and game tree search

**Your contribution**: Bridge this gap by:
1. Formalizing quasi-probabilities in MCTS context
2. Showing practical benefits (faster search, better pruning)
3. Connecting Feynman's quasi-probabilities to game tree search

---

## Experimental Protocol

### Research Questions

**RQ1**: Do quasi-probabilities improve search efficiency?
- **Hypothesis**: QP-MCTS explores fewer nodes to find optimal moves
- **Test**: Compare nodes explored on tactical puzzles
- **Success**: >20% reduction in nodes

**RQ2**: Does explicit negative representation enable better pruning?
- **Hypothesis**: Negative probabilities identify bad branches faster
- **Test**: Measure pruning rate and accuracy
- **Success**: Higher pruning rate without quality loss

**RQ3**: Can neural networks learn to predict quasi-probabilities?
- **Hypothesis**: Q_positive and Q_negative are learnable
- **Test**: Training curves, prediction accuracy
- **Success**: MSE < baseline within 50% of training time

**RQ4**: Does QP-MCTS scale to complex games?
- **Hypothesis**: Benefits increase with branching factor
- **Test**: Compare on tic-tac-toe, chess, Go
- **Success**: Improvement scales with complexity

### Controlled Experiments

#### Experiment 1: Toy Problem (Tic-Tac-Toe)

**Setup**:
```
Algorithms: Standard MCTS, QP-MCTS
Simulations: 10, 50, 100, 500
Positions: All possible tic-tac-toe positions
Repetitions: 10 runs per configuration
```

**Measurements**:
- Nodes explored per search
- Optimal move found? (yes/no)
- Time per search
- Tree depth reached

**Expected results**:
- QP-MCTS finds optimal moves with fewer nodes
- Difference most pronounced at low simulation counts

#### Experiment 2: Chess Tactics

**Setup**:
```
Algorithms: Standard MCTS, QP-MCTS, AlphaZero, QP-AlphaZero
Dataset: 1000 tactical puzzles (mate-in-2, mate-in-3, etc.)
Simulations: 100, 500, 1000, 2000
Hardware: Same GPU for all
```

**Measurements**:
- Solution rate (% puzzles solved)
- Average nodes to solution
- Average time to solution
- False pruning rate (pruned winning moves)

**Expected results**:
- QP variants solve same puzzles with fewer nodes
- Particularly strong on deep tactics (mate-in-5+)

#### Experiment 3: Full Game Play

**Setup**:
```
Tournament: Round-robin between all algorithms
Games: 100 games per pairing
Time control: 1 second per move
Opening book: First 4 moves randomized
```

**Measurements**:
- Win/draw/loss record
- Elo rating (computed from results)
- Average move quality (Stockfish eval)
- Computational efficiency (nodes/second)

**Expected results**:
- QP variants achieve similar Elo with lower computational cost
- Or: same computational budget, higher Elo

### Ablation Studies

**Ablation 1: Quasi-Probability Components**

Test each component independently:
- Only Q_positive (no negative tracking)
- Only Q_negative (no positive tracking)
- Full Q_quasi (both components)
- Different combinations in UCB formula

**Ablation 2: Normalization Schemes**

Test different normalization approaches:
- No normalization
- Normalize over siblings
- Normalize over entire tree
- Adaptive normalization

**Ablation 3: Backpropagation Rules**

Test different update rules:
- Standard averaging
- Exponential moving average
- Separate learning rates for Q_pos/Q_neg
- Optimistic vs. pessimistic updates

**Ablation 4: Network Architecture**

If using neural networks:
- Single head (standard value)
- Dual heads (Q_pos + Q_neg)
- Triple heads (policy + value + quasi-prob)
- Shared vs. separate layers

---

## Publication Strategy

### Target Venues

#### Tier 1 (Top ML Conferences)

**NeurIPS (Neural Information Processing Systems)**
- Deadline: May
- Focus: Novel algorithms, strong theoretical contribution
- What they want: Significant empirical improvement + theory
- Strategy: Submit if QP-AlphaZero shows clear wins

**ICML (International Conference on Machine Learning)**
- Deadline: January
- Focus: Machine learning theory and applications
- What they want: Rigorous experiments, ablations
- Strategy: Submit if have strong empirical results

**ICLR (International Conference on Learning Representations)**
- Deadline: September
- Focus: Deep learning, representation learning
- What they want: Novel representations, strong benchmarks
- Strategy: Submit if neural network component is strong

#### Tier 2 (Strong Conferences)

**AAAI (Association for Advancement of AI)**
- Deadline: August
- Focus: Broader AI, includes search and planning
- What they want: Solid work, good experiments
- Strategy: Good backup option

**IJCAI (International Joint Conference on AI)**
- Deadline: January
- Focus: All AI topics
- What they want: Novel techniques, applications
- Strategy: Reliable venue

**CoG (Conference on Games)**
- Deadline: April
- Focus: Games, MCTS, game-playing AI
- What they want: Game-specific contributions
- Strategy: Best fit if focused on chess/Go

#### Tier 3 (Workshops, Specialized Venues)

**NeurIPS Workshop on Planning**
- Good for preliminary results
- Fast turnaround
- Can get feedback before main conference

**ICML Workshop on Reinforcement Learning**
- If RL framing is strong
- Less competitive
- Good for novel ideas

### Paper Structure

#### Title Options

1. "Quasi-Probability Monte Carlo Tree Search" (descriptive)
2. "Negative Probabilities for Game Tree Search" (provocative)
3. "Backward Reasoning in Monte Carlo Tree Search via Quasi-Probabilities" (explanatory)
4. "Refining Search with Negative Values: A Quasi-Probability Approach to MCTS" (technical)

**Recommendation**: Option 3 - clear and accurate

#### Abstract Template

```
Monte Carlo Tree Search (MCTS) has achieved remarkable success in
game-playing AI through forward simulation and value backpropagation.
However, standard MCTS treats all unexplored branches symmetrically,
lacking explicit representation of refuted or unlikely paths. We
introduce Quasi-Probability MCTS (QP-MCTS), which maintains both
positive probabilities for promising paths and negative probabilities
for refuted paths, inspired by quasi-probability distributions in
quantum mechanics. Our method extends MCTS's backpropagation phase to
separately track path promise and path refutation, enabling more
aggressive pruning while maintaining correctness. Experiments on
tic-tac-toe, chess, and Go show QP-MCTS achieves [X%] reduction in
nodes explored compared to standard MCTS, with comparable or superior
move quality. When combined with neural networks (QP-AlphaZero), our
approach learns to predict quasi-probabilities from self-play, further
improving search efficiency. We provide theoretical analysis of
convergence properties and empirically demonstrate that explicit
negative representation accelerates both search and learning.
```

#### Section Outline

**1. Introduction** (1.5 pages)
- Motivation: MCTS success but lacks explicit refutation
- Core idea: Quasi-probabilities from physics
- Contributions: Algorithm, theory, experiments
- Results preview: X% fewer nodes, Y% faster

**2. Background** (2 pages)
- MCTS overview and notation
- UCB and backpropagation
- AlphaZero architecture
- Quasi-probabilities in quantum mechanics

**3. Quasi-Probability MCTS** (3 pages)
- Node structure with Q_pos/Q_neg
- Modified selection rule
- Backpropagation with dual updates
- Normalization to maintain distribution
- Pseudocode

**4. Theoretical Properties** (2 pages)
- Convergence guarantees
- Pruning correctness
- Relationship to standard MCTS
- Computational complexity

**5. Neural Network Extension** (2 pages)
- QP-Network architecture
- Training with self-play
- Loss function design
- Integration with MCTS

**6. Experiments** (4 pages)
- Setup: environments, baselines, metrics
- Results on tactical puzzles
- Full game playing
- Ablation studies
- Scaling analysis

**7. Related Work** (1.5 pages)
- MCTS variants
- Tree search improvements
- Quasi-probabilities in ML
- Negative representations

**8. Discussion** (1 page)
- When QP-MCTS helps most
- Limitations
- Future work

**9. Conclusion** (0.5 pages)
- Summary of contributions
- Broader impact

**Total**: ~18 pages (typical for NeurIPS/ICML)

#### Key Figures to Include

**Figure 1**: QP-MCTS tree visualization
- Show same position with standard MCTS vs QP-MCTS
- Color-code nodes by Q_positive (green), Q_negative (red)
- Highlight pruned branches

**Figure 2**: Node efficiency comparison
- X-axis: Number of simulations
- Y-axis: Solution rate on tactical puzzles
- Lines: Standard MCTS, QP-MCTS, AlphaZero, QP-AlphaZero

**Figure 3**: Pruning analysis
- Show trajectory of Q_negative for good vs bad moves
- Demonstrate bad moves get high Q_negative quickly

**Figure 4**: Neural network learning curves
- Training loss over self-play iterations
- Compare convergence speed: Standard vs QP

**Figure 5**: Scaling with game complexity
- Show benefit increases with branching factor
- Test on tic-tac-toe, chess, Go

**Figure 6**: Ablation study results
- Bar charts showing contribution of each component

### Success Criteria for Publication

**Minimum bar (workshop paper)**:
- ✓ Clear algorithm description
- ✓ Toy problem showing proof of concept
- ✓ Some theoretical insight

**Conference paper (AAAI/IJCAI)**:
- ✓ Significant improvement on chess tactics (>20% fewer nodes)
- ✓ Ablation studies showing which components matter
- ✓ Theoretical analysis of convergence

**Top-tier venue (NeurIPS/ICML)**:
- ✓ Major improvement on full game play (>50 Elo or >30% nodes)
- ✓ Strong theoretical contribution (proofs)
- ✓ Neural network extension working
- ✓ Scales to Go or other complex games

---

## Code Architecture

### Repository Structure

```
qp-mcts/
├── README.md
├── requirements.txt
├── setup.py
│
├── qp_mcts/                    # Main package
│   ├── __init__.py
│   ├── node.py                 # QPNode class
│   ├── mcts.py                 # Standard MCTS
│   ├── qp_mcts.py              # QP-MCTS algorithm
│   ├── ucb.py                  # UCB variants
│   └── utils.py
│
├── environments/               # Game implementations
│   ├── __init__.py
│   ├── tictactoe.py
│   ├── chess_env.py
│   └── go_env.py
│
├── neural/                     # Neural network components
│   ├── __init__.py
│   ├── networks.py             # QPNetwork, AlphaZero
│   ├── training.py             # Self-play loop
│   └── evaluation.py
│
├── experiments/                # Experiment scripts
│   ├── toy_problem.py          # Tic-tac-toe tests
│   ├── chess_tactics.py        # Puzzle evaluation
│   ├── full_games.py           # Game playing
│   └── ablations.py
│
├── visualization/              # Plotting and analysis
│   ├── tree_viz.py
│   ├── learning_curves.py
│   └── statistics.py
│
├── tests/                      # Unit tests
│   ├── test_node.py
│   ├── test_mcts.py
│   └── test_qp_mcts.py
│
└── paper/                      # LaTeX paper
    ├── main.tex
    ├── figures/
    └── references.bib
```

### Core Classes

```python
# qp_mcts/node.py

class QPNode:
    """Node in quasi-probability MCTS tree."""

    def __init__(self, state, parent=None, action=None, prior=0.0):
        # State information
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node

        # Standard MCTS
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}  # action → QPNode

        # Quasi-probability components
        self.Q_positive = 0.0    # Promise of this path
        self.Q_negative = 0.0    # Refutation of this path
        self.Q_quasi = 0.0       # Combined quasi-probability

        # Neural network prior
        self.prior = prior

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        # Check if game is over
        pass

    def expand(self, action_priors):
        """Add children for all legal actions."""
        for action, prior in action_priors:
            if action not in self.children:
                child = QPNode(
                    state=self.state.make_move(action),
                    parent=self,
                    action=action,
                    prior=prior
                )
                self.children[action] = child

    def select_child(self, c_puct=1.0, use_quasi=True):
        """Select best child using UCB."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            if use_quasi:
                score = self._ucb_quasi(child, c_puct)
            else:
                score = self._ucb_standard(child, c_puct)

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _ucb_quasi(self, child, c_puct):
        """UCB with quasi-probabilities."""
        if child.visits == 0:
            return float('inf')

        # Exploitation: use quasi-probability
        exploitation = child.Q_quasi

        # Exploration: standard UCB
        exploration = c_puct * child.prior * \
                     np.sqrt(self.visits) / (1 + child.visits)

        # Refutation penalty: avoid refuted paths
        refutation = -child.Q_negative * 0.5

        return exploitation + exploration + refutation

    def _ucb_standard(self, child, c_puct):
        """Standard UCB for comparison."""
        if child.visits == 0:
            return float('inf')

        q_value = child.value_sum / child.visits
        u_value = c_puct * child.prior * \
                 np.sqrt(self.visits) / (1 + child.visits)

        return q_value + u_value

    def update(self, value):
        """Update node with new value."""
        self.visits += 1
        self.value_sum += value

        # Update quasi-probability components
        alpha = 0.1  # Learning rate

        if value > 0:
            # Good outcome
            self.Q_positive += alpha * (1.0 - self.Q_positive)
        else:
            # Bad outcome
            self.Q_negative += alpha * (1.0 - self.Q_negative)

        # Compute quasi-probability
        self.Q_quasi = self.Q_positive - self.Q_negative

    def get_normalized_quasi_prob(self):
        """Get normalized quasi-probability over siblings."""
        if self.parent is None:
            return 1.0

        siblings = list(self.parent.children.values())
        total_abs = sum(abs(s.Q_quasi) for s in siblings)

        if total_abs > 0:
            return self.Q_quasi / total_abs
        else:
            return 0.0
```

```python
# qp_mcts/qp_mcts.py

class QPMCTS:
    """Quasi-Probability Monte Carlo Tree Search."""

    def __init__(self, game, network=None, num_simulations=800,
                 c_puct=1.0, use_quasi=True):
        self.game = game
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.use_quasi = use_quasi

        self.root = None
        self.total_nodes = 0

    def search(self, state):
        """Run MCTS from given state."""
        self.root = QPNode(state)

        # If we have a network, expand root immediately
        if self.network is not None:
            self._expand_node(self.root)

        for _ in range(self.num_simulations):
            # Selection
            node = self._select(self.root)

            # Expansion
            if not node.is_terminal():
                if node.is_leaf() and node.visits > 0:
                    self._expand_node(node)
                    # Select one of the children
                    if node.children:
                        _, node = node.select_child(self.c_puct, self.use_quasi)

            # Evaluation
            value = self._evaluate(node)

            # Backpropagation
            self._backpropagate(node, value)

    def _select(self, node):
        """Walk down tree using UCB."""
        while not node.is_leaf() and not node.is_terminal():
            _, node = node.select_child(self.c_puct, self.use_quasi)
        return node

    def _expand_node(self, node):
        """Expand node by adding children."""
        if self.network is not None:
            # Get policy and value from network
            policy, value = self.network.predict(node.state)
            action_priors = list(zip(
                self.game.legal_actions(node.state),
                policy
            ))
        else:
            # Uniform prior
            actions = self.game.legal_actions(node.state)
            uniform_prob = 1.0 / len(actions) if actions else 0
            action_priors = [(a, uniform_prob) for a in actions]

        node.expand(action_priors)
        self.total_nodes += len(node.children)

    def _evaluate(self, node):
        """Evaluate leaf node."""
        if node.is_terminal():
            return self.game.get_value(node.state)

        if self.network is not None:
            _, value = self.network.predict(node.state)
            return value
        else:
            # Random rollout
            return self._rollout(node.state)

    def _rollout(self, state):
        """Random simulation from state to terminal."""
        current_state = state.copy()

        while not self.game.is_terminal(current_state):
            actions = self.game.legal_actions(current_state)
            action = random.choice(actions)
            current_state = self.game.make_move(current_state, action)

        return self.game.get_value(current_state)

    def _backpropagate(self, node, value):
        """Update values up to root."""
        current = node

        while current is not None:
            current.update(value)
            value = -value  # Flip for opponent
            current = current.parent

        # Normalize quasi-probabilities at each level
        if self.use_quasi:
            self._normalize_tree(self.root)

    def _normalize_tree(self, node):
        """Normalize quasi-probabilities."""
        if not node.children:
            return

        # Normalize over children
        children = list(node.children.values())
        total_abs = sum(abs(c.Q_quasi) for c in children)

        if total_abs > 0:
            for child in children:
                child.Q_quasi /= total_abs
                # Update components
                if child.Q_quasi >= 0:
                    child.Q_positive = child.Q_quasi
                    child.Q_negative = 0
                else:
                    child.Q_positive = 0
                    child.Q_negative = -child.Q_quasi

        # Recurse
        for child in children:
            self._normalize_tree(child)

    def get_action_probs(self, temperature=1.0):
        """Get action probabilities from root."""
        if not self.root.children:
            return {}

        actions = []
        visits = []

        for action, child in self.root.children.items():
            actions.append(action)
            visits.append(child.visits)

        visits = np.array(visits)

        if temperature == 0:
            # Deterministic: pick most visited
            probs = np.zeros_like(visits, dtype=float)
            probs[np.argmax(visits)] = 1.0
        else:
            # Boltzmann distribution
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)

        return dict(zip(actions, probs))

    def get_best_action(self):
        """Get action with highest visit count."""
        if not self.root.children:
            return None

        return max(
            self.root.children.items(),
            key=lambda item: item[1].visits
        )[0]
```

### Integration Points

**With Python-Chess**:
```python
# environments/chess_env.py

import chess
from qp_mcts import QPMCTS

class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def legal_actions(self, state):
        return list(state.legal_moves)

    def make_move(self, state, action):
        new_state = state.copy()
        new_state.push(action)
        return new_state

    def is_terminal(self, state):
        return state.is_game_over()

    def get_value(self, state):
        if state.is_checkmate():
            return -1  # Current player lost
        else:
            return 0   # Draw

# Usage
game = ChessGame()
mcts = QPMCTS(game, num_simulations=1000, use_quasi=True)

board = chess.Board()
mcts.search(board)
best_move = mcts.get_best_action()
```

**With Neural Network**:
```python
# neural/training.py

def train_qp_network():
    game = ChessGame()
    network = QPNetwork()

    for iteration in range(1000):
        # Self-play
        games = []
        for _ in range(100):
            game_data = self_play_game(game, network)
            games.extend(game_data)

        # Train
        train_network(network, games)

        # Evaluate
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: "
                  f"Loss = {evaluate(network)}")

def self_play_game(game, network):
    mcts = QPMCTS(game, network, num_simulations=800)

    board = chess.Board()
    game_data = []

    while not board.is_game_over():
        mcts.search(board)
        pi = mcts.get_action_probs()

        # Store (state, policy, qp_components)
        qp_data = mcts.root.get_quasi_prob_data()
        game_data.append((board.copy(), pi, qp_data))

        # Make move
        action = sample(pi)
        board.push(action)

    # Label with outcome
    z = get_outcome(board)
    return [(s, p, z, qp) for (s, p, qp) in game_data]
```

---

## Timeline and Milestones

### Month 1: Foundation

**Week 1-2: Implementation**
- [ ] Implement QPNode class
- [ ] Implement basic QP-MCTS algorithm
- [ ] Write unit tests
- [ ] Set up tic-tac-toe environment

**Week 3: Validation**
- [ ] Test on tic-tac-toe
- [ ] Compare against standard MCTS
- [ ] Verify correctness (optimal play)
- [ ] Measure node efficiency

**Week 4: Analysis**
- [ ] Visualize trees
- [ ] Analyze quasi-probability distributions
- [ ] Write up preliminary results
- [ ] Identify issues/improvements

**Milestone 1**: Working QP-MCTS on toy problem

### Month 2: Chess

**Week 5-6: Chess Integration**
- [ ] Integrate with python-chess
- [ ] Implement chess-specific evaluation
- [ ] Set up tactical puzzle dataset
- [ ] Run preliminary experiments

**Week 7: Optimization**
- [ ] Profile code, identify bottlenecks
- [ ] Optimize hot paths
- [ ] Parallelize MCTS (if needed)
- [ ] Test at scale (1000+ puzzles)

**Week 8: Comprehensive Evaluation**
- [ ] Full tactical puzzle benchmark
- [ ] Play games vs Stockfish
- [ ] Analyze results
- [ ] Ablation studies

**Milestone 2**: QP-MCTS competitive on chess

### Month 3: Neural Networks

**Week 9-10: Network Design**
- [ ] Design QPNetwork architecture
- [ ] Implement training loop
- [ ] Set up self-play pipeline
- [ ] Debug training

**Week 11-12: Training**
- [ ] Run self-play training
- [ ] Monitor convergence
- [ ] Tune hyperparameters
- [ ] Evaluate learned network

**Milestone 3**: Working QP-AlphaZero

### Month 4: Experiments & Writing

**Week 13: Final Experiments**
- [ ] Run all benchmarks
- [ ] Collect statistics
- [ ] Generate all figures
- [ ] Statistical significance tests

**Week 14-15: Paper Writing**
- [ ] Write first draft
- [ ] Create figures/tables
- [ ] Internal review
- [ ] Revisions

**Week 16: Submission**
- [ ] Final experiments if needed
- [ ] Polish paper
- [ ] Submit to conference
- [ ] Prepare code release

**Milestone 4**: Paper submitted!

### Post-Submission

**Ongoing:**
- [ ] Respond to reviews
- [ ] Additional experiments if requested
- [ ] Prepare rebuttal
- [ ] Camera-ready version

---

## Key Advantages of This Direction

### 1. You Have a Simulator

**Unlike model-free RL**: In chess/Go, you know the rules perfectly. You can simulate any move sequence. This eliminates the core problem with counterfactual reasoning - you're not guessing what would happen, you can actually check.

### 2. Strong Baselines Exist

You can compare against:
- Standard MCTS (well-studied)
- AlphaZero (state-of-the-art)
- Stockfish (superhuman chess engine)

This makes evaluation clear and convincing.

### 3. Theory is Cleaner

MCTS has solid theoretical foundations:
- Convergence guarantees
- Regret bounds
- Connection to bandits

You can build on this to prove properties of QP-MCTS.

### 4. Immediate Applications

Game-playing AI is high-impact:
- Chess engines used by millions
- AlphaGo was a major milestone
- Game-playing is a standard benchmark

Success here gets attention.

### 5. Scalability Path

Start simple, scale up:
- Tic-tac-toe (days)
- Chess (weeks)
- Go (months)
- General games (future work)

This gives a clear research trajectory.

---

## Potential Challenges & Solutions

### Challenge 1: No Improvement Over MCTS

**Risk**: QP-MCTS doesn't actually help.

**Mitigation**:
- Start with careful theoretical analysis
- Prove that negative values CAN help in specific scenarios
- Design experiments that highlight those scenarios
- Even null result is publishable if rigorous

**Fallback**: Paper on "Why Quasi-Probabilities Don't Help in Tree Search: An Analysis"

### Challenge 2: Neural Network Training is Hard

**Risk**: QP-AlphaZero doesn't converge or doesn't improve.

**Mitigation**:
- Start with simpler supervised learning (predict from MCTS trees)
- Use smaller networks initially
- Careful hyperparameter tuning
- Compare against strong baseline (standard AlphaZero)

**Fallback**: Focus on non-neural QP-MCTS results

### Challenge 3: Computation Time

**Risk**: QP-MCTS is slower than standard MCTS due to overhead.

**Mitigation**:
- Optimize implementation carefully
- Use efficient data structures
- Profile and improve hot paths
- Compare nodes explored, not wall-clock time

**Alternative metric**: Nodes to solution (if QP uses fewer nodes, even if slower per node, it's still progress)

### Challenge 4: Negative Values Become Noisy

**Risk**: Q_negative estimates are unreliable, hurt performance.

**Mitigation**:
- Use higher confidence threshold before pruning
- Implement conservative pruning (only prune if very confident)
- Adaptive pruning based on tree statistics
- Ablation study on pruning aggressiveness

### Challenge 5: Doesn't Scale Beyond Chess

**Risk**: Works on chess but not Go or other games.

**Mitigation**:
- Carefully analyze why it works on chess
- Identify properties that determine success
- Test on intermediate complexity games
- Be honest about limitations

**Framing**: "QP-MCTS is particularly effective for games with properties X, Y, Z"

---

## Broader Impact

### Connections to Other Fields

**Quantum Computing**:
- Quasi-probabilities are foundational in quantum mechanics
- Your work provides ML interpretation
- Could inspire quantum ML algorithms

**Decision Theory**:
- Humans reason about "what not to do" (negative evidence)
- QP-MCTS formalizes this
- Applications to human-AI collaboration

**Planning & Robotics**:
- Explicit "don't go there" signals useful for safety
- Negative values could represent dangerous states
- Transfer to motion planning

**Cognitive Science**:
- Do humans use "negative probabilities" implicitly?
- QP-MCTS as cognitive model
- Understanding human planning strategies

### Ethical Considerations

**Positive aspects**:
- Better game-playing AI (education, entertainment)
- More efficient algorithms (lower compute, less energy)
- Theoretical contributions to AI safety (explicit danger representation)

**Risks to consider**:
- Stronger game AI could reduce human interest (minor)
- Dual use in adversarial domains (game-theoretic AI)

**Overall**: Low risk, high educational value.

---

## Resources Needed

### Computational Resources

**For tic-tac-toe**: Your laptop (minutes)

**For chess experiments**:
- Modern CPU sufficient for tactics puzzles
- GPU helpful for neural network training
- Estimate: 1 week on 1 GPU for basic training

**For Go experiments**:
- More compute needed (weeks on 1-4 GPUs)
- Can start with smaller board sizes (9x9, 13x13)

**Total estimate**: If you have access to a GPU, enough compute available.

### Software Dependencies

```python
# requirements.txt

numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0

# Chess
python-chess>=1.9.0

# Deep learning (optional, for Phase 3)
torch>=1.9.0
torchvision>=0.10.0

# Utilities
tqdm>=4.62.0
pandas>=1.3.0
seaborn>=0.11.0

# Testing
pytest>=6.2.0
```

### Knowledge Requirements

**What you need to know**:
- Tree search algorithms (you can learn)
- Basic game theory (you can learn)
- Python programming (you have)
- RL fundamentals (you have)

**What's helpful but not required**:
- Chess (can use engine evaluations)
- Neural networks (start without them)
- Quantum mechanics (just inspiration)

### Time Commitment

**Part-time** (10-15 hours/week):
- 4 months to first submission
- 6 months to complete paper

**Full-time** (40+ hours/week):
- 6 weeks to first submission
- 3 months to complete paper

---

## Why This Will Work

### Your Original Insight Was Correct

You wanted **backward reasoning and counterfactuals**. MCTS already does this! You just need to make it explicit with quasi-probabilities.

### Clear Success Criteria

Unlike model-free RL where "does it help?" is murky, here it's clear:
- Fewer nodes to find optimal move? ✓ or ✗
- Faster search? ✓ or ✗
- Better pruning? ✓ or ✗

### Multiple Publication Paths

Even if main result is weak:
- Theoretical contribution (novel framework)
- Negative result (why it doesn't help)
- Specific domain success (chess tactics only)
- Foundation for future work

All are publishable with right framing.

### Builds on Solid Foundation

Not inventing new RL algorithm from scratch. Extending proven method (MCTS) with principled enhancement (quasi-probabilities).

### Natural Scaling Path

Start tiny (tic-tac-toe), scale incrementally. Each step builds confidence and gives publishable results.

---

## Concrete First Steps

### This Week

**Day 1-2**: Read MCTS survey paper
- Understand UCB algorithm
- Study backpropagation phase
- Identify where quasi-probabilities fit

**Day 3-4**: Implement basic MCTS on tic-tac-toe
- Get standard version working
- Verify optimal play
- Understand codebase

**Day 5-7**: Add quasi-probability components
- Extend node with Q_pos/Q_neg
- Modify backpropagation
- Test on tic-tac-toe

**Deliverable**: Working QP-MCTS on tic-tac-toe with visualization

### Next Week

**Day 8-10**: Comprehensive testing
- Compare node counts
- Verify correctness
- Generate figures

**Day 11-12**: Analysis and writeup
- Why does it work (or not)?
- What scenarios benefit most?
- Prepare preliminary report

**Day 13-14**: Plan chess extension
- Set up python-chess
- Design evaluation framework
- Identify tactical puzzle datasets

**Deliverable**: Report on tic-tac-toe results, plan for chess

---

## Conclusion

This direction is **much more promising** than model-free RL in grid worlds because:

1. **You have perfect simulators** (game rules)
2. **MCTS already does backward reasoning** (you're enhancing it)
3. **Clear evaluation metrics** (nodes explored, move quality)
4. **Strong baselines** (standard MCTS, AlphaZero)
5. **Solid theory** (can prove properties)
6. **Natural applications** (chess, Go, games)

Your core insight about **counterfactual reasoning and backward propagation** is valuable. It just needs the right setting - and MCTS provides that setting.

Start with tic-tac-toe to validate the concept (1-2 weeks), then scale to chess (1-2 months), then optionally add neural networks (1-2 months). You'll have multiple publication opportunities along the way.

**The path is clear. The theory is sound. The experiments are feasible. Time to build it!**
