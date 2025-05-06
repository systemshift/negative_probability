Quasi‑Probability Reinforcement Learning (QP‑RL)

Bidirectional‑time RL where positive probabilities model future transitions and negative quasi‑probabilities model plausible past histories of the current state.  This yields an agent that can rewind to alternative timelines, explore them, then resume normal forward learning—all within one unified value table.

Key Ideas

Concept

Explanation

Quasi‑Probability Table

For every state–action pair  store a real number .   = likelihood of moving forward to the successor;  = plausibility that  lay on a past trajectory to the current state.  (

q

=1) marks logically impossible paths.

Bidirectional Sampling

At each step the agent draws from the offset‑shifted distribution of  values.  Positive draws move forward; negative draws trigger a jump to the chosen past state.

Learning Rule

After every transition update two entries: one positive (future) and one negative (past) so that reachable paths drift toward  while impossible paths saturate at .

Minimal Pseudocode

# q: defaultdict(lambda: defaultdict(float))
for episode in range(E):
    s = env.reset(); traj = []
    for t in range(T):
        a = sample_qp(q[s])              # shift–softmax over positive+negative
        s2, r, done, _ = env.step(a)
        traj.append((s, a, r))

        # forward update (future)
        q[s][a] += alpha * (r - q[s][a])

        # backward update (most recent past action)
        if len(traj) > 1:
            sp, ap, _ = traj[-2]
            q[sp][ap] -= alpha * (abs(q[sp][ap]) - eps)

        s = s2
        if done:
            break

Demo Environment Suggestion

Timeline‑Grid – 5×5 grid with one‑way doors and portals.  Impossible pasts flagged at ; feasible but sub‑optimal pasts start near .  Visualise with a heat‑map: blue = negative (past), red = positive (future).

Why It Matters

Efficient Exploration: the agent revisits only plausible alternative histories rather than resetting the whole episode (contrast with HER).

Unified Representation: no separate memory buffer—past and future encoded in the same table.

Generic: works with tabular Q‑learning, DQN, or tree search (replace counts with quasi‑probabilities).

