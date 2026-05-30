"""REINFORCE on a 5x5 gridworld.

The worked example for the closing post of the inductive-bias series.
NumPy, not pure Python: a policy-gradient run does thousands of episodes,
each up to 25 steps, and the per-step Python overhead of the scratchnn
core would dominate the actual learning. The math is the same: a small
MLP with one hidden layer, softmax output over four discrete actions,
trained by gradient ascent on a return-weighted log-likelihood.

Setup:
  - 5x5 grid. Start (0, 0). Goal (4, 4).
  - Actions: 0=up, 1=down, 2=left, 3=right. Off-grid moves are no-ops.
  - Reward: -0.01 per step + 1.0 on reaching the goal.
  - Discount gamma = 0.99. Max episode length 25.
  - Policy: MLP with one hidden layer of 64 tanh units, softmax over
    4 actions. 25-dim one-hot state input.
  - Update: REINFORCE with a moving-average return baseline.
  - Optimizer: plain SGD on the policy parameters.

Run: python examples/reinforce_gridworld.py
"""
import numpy as np


# ============================================================
# Environment
# ============================================================

GRID = 5
N_STATES = GRID * GRID
N_ACTIONS = 4  # up, down, left, right
START = (0, 0)
GOAL = (GRID - 1, GRID - 1)
STEP_REWARD = -0.01
GOAL_REWARD = 1.0
MAX_STEPS = 25
GAMMA = 0.99


def state_index(rc):
    r, c = rc
    return r * GRID + c


def one_hot(idx, n=N_STATES):
    v = np.zeros(n, dtype=np.float64)
    v[idx] = 1.0
    return v


def step(rc, a):
    """Apply action a to position rc. Off-grid moves are no-ops."""
    r, c = rc
    if a == 0:    # up
        r = max(0, r - 1)
    elif a == 1:  # down
        r = min(GRID - 1, r + 1)
    elif a == 2:  # left
        c = max(0, c - 1)
    elif a == 3:  # right
        c = min(GRID - 1, c + 1)
    new = (r, c)
    if new == GOAL:
        return new, GOAL_REWARD, True
    return new, STEP_REWARD, False


# ============================================================
# Policy network
# ============================================================

class PolicyMLP:
    """One hidden layer, tanh, then linear logits over 4 actions.

    The forward pass mirrors the scratchnn Linear + Tanh + Linear stack
    exactly, just vectorized in NumPy. We retain per-step activations so
    that backward can run after the episode.
    """

    def __init__(self, n_in=N_STATES, n_hidden=64, n_out=N_ACTIONS, seed=0):
        rng = np.random.default_rng(seed)
        # He-ish init for tanh, scaled down a touch.
        self.W1 = rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        """x is one-hot state of shape (N_STATES,)."""
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2 + self.b2
        return z2, h

    def probs(self, x):
        z, _ = self.forward(x)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def grads_for_episode(self, states, actions, weights):
        """Return per-parameter gradients of -sum_t w_t * log pi(a_t|s_t).

        The minus sign is so that *descending* on these gradients ascends
        on expected return. states is (T, N_STATES), actions is (T,),
        weights is (T,) and carries the return-to-go minus baseline.
        """
        T = len(actions)
        # Forward pass for the whole episode.
        Z1 = states @ self.W1 + self.b1
        H = np.tanh(Z1)
        Z2 = H @ self.W2 + self.b2
        Z2 = Z2 - Z2.max(axis=1, keepdims=True)
        E = np.exp(Z2)
        P = E / E.sum(axis=1, keepdims=True)

        # Backward through softmax cross-entropy, exactly p - onehot(a)
        # at each row, then scaled by the per-step weight.
        dlogits = P.copy()
        dlogits[np.arange(T), actions] -= 1.0
        dlogits *= weights[:, None]

        dW2 = H.T @ dlogits
        db2 = dlogits.sum(axis=0)

        dH = dlogits @ self.W2.T
        dZ1 = dH * (1.0 - H * H)

        dW1 = states.T @ dZ1
        db1 = dZ1.sum(axis=0)

        return [dW1, db1, dW2, db2]

    def step_sgd(self, grads, lr):
        for p, g in zip(self.params(), grads):
            p -= lr * g


# ============================================================
# REINFORCE loop
# ============================================================

def rollout(policy, rng):
    """Run one episode under the current policy. Return arrays of
    states (one-hot), actions, rewards, and the total return."""
    rc = START
    states = []
    actions = []
    rewards = []
    for _ in range(MAX_STEPS):
        s_idx = state_index(rc)
        x = one_hot(s_idx)
        p = policy.probs(x)
        a = int(rng.choice(N_ACTIONS, p=p))
        rc, r, done = step(rc, a)
        states.append(x)
        actions.append(a)
        rewards.append(r)
        if done:
            break
    return (np.array(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float64))


def returns_to_go(rewards, gamma=GAMMA):
    """Compute G_t = sum_{k>=0} gamma^k r_{t+k+1}."""
    G = np.zeros_like(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        G[t] = acc
    return G


def train(n_episodes=2000, lr=0.05, seed=0, log_every=100, baseline_beta=0.9):
    rng = np.random.default_rng(seed)
    policy = PolicyMLP(seed=seed)

    history = []           # raw episode returns
    smoothed = []          # smoothed return per log_every
    baseline = 0.0         # moving-average baseline for variance reduction

    for ep in range(n_episodes):
        states, actions, rewards = rollout(policy, rng)
        G = returns_to_go(rewards)
        episode_return = float(rewards.sum())

        # Variance-reduced weights: subtract a state-independent baseline.
        # The expectation of the score function times any state-independent
        # quantity is zero, so this is unbiased.
        baseline = baseline_beta * baseline + (1 - baseline_beta) * episode_return
        weights = G - baseline

        grads = policy.grads_for_episode(states, actions, weights)
        policy.step_sgd(grads, lr)

        history.append(episode_return)

        if (ep + 1) % log_every == 0:
            recent = np.mean(history[-log_every:])
            smoothed.append((ep + 1, recent))
            print(f"episode {ep+1:5d}  mean return (last {log_every}): "
                  f"{recent:+.4f}")

    return policy, history, smoothed


# ============================================================
# Evaluation
# ============================================================

def greedy_trajectory(policy):
    """Roll out an episode picking the argmax action each step.

    Returns the list of visited cells and total reward.
    """
    rc = START
    cells = [rc]
    total = 0.0
    for _ in range(MAX_STEPS):
        x = one_hot(state_index(rc))
        p = policy.probs(x)
        a = int(np.argmax(p))
        rc, r, done = step(rc, a)
        cells.append(rc)
        total += r
        if done:
            break
    return cells, total


def sample_trajectory(policy, rng):
    rc = START
    cells = [rc]
    total = 0.0
    for _ in range(MAX_STEPS):
        x = one_hot(state_index(rc))
        p = policy.probs(x)
        a = int(rng.choice(N_ACTIONS, p=p))
        rc, r, done = step(rc, a)
        cells.append(rc)
        total += r
        if done:
            break
    return cells, total


def evaluate(policy, n_eval=200, seed=999):
    rng = np.random.default_rng(seed)
    returns = []
    lengths = []
    reached = 0
    for _ in range(n_eval):
        cells, total = sample_trajectory(policy, rng)
        returns.append(total)
        lengths.append(len(cells) - 1)
        if cells[-1] == GOAL:
            reached += 1
    return {
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": reached / n_eval,
    }


def main():
    print("REINFORCE on 5x5 gridworld")
    print(f"  grid {GRID}x{GRID}, start {START}, goal {GOAL}")
    print(f"  reward {STEP_REWARD}/step + {GOAL_REWARD} at goal")
    print(f"  gamma {GAMMA}, max steps {MAX_STEPS}")
    print()

    # Pre-training evaluation under the untrained policy.
    init_policy = PolicyMLP(seed=0)
    pre = evaluate(init_policy)
    print("Before training:")
    print(f"  mean return  {pre['mean_return']:+.4f}")
    print(f"  mean length  {pre['mean_length']:5.2f} steps")
    print(f"  success rate {pre['success_rate']:.1%}")
    print()

    # Show a sampled trajectory under the untrained policy.
    rng = np.random.default_rng(42)
    cells, total = sample_trajectory(init_policy, rng)
    print(f"  sample trajectory ({len(cells)-1} steps, return {total:+.4f}):")
    print(f"    {cells}")
    print()

    # Train.
    policy, history, smoothed = train(n_episodes=2000, lr=0.05, seed=0)
    print()

    # Post-training evaluation.
    post = evaluate(policy)
    print("After training:")
    print(f"  mean return  {post['mean_return']:+.4f}")
    print(f"  mean length  {post['mean_length']:5.2f} steps")
    print(f"  success rate {post['success_rate']:.1%}")
    print()

    # Greedy trajectory.
    cells, total = greedy_trajectory(policy)
    print(f"  greedy trajectory ({len(cells)-1} steps, return {total:+.4f}):")
    print(f"    {cells}")
    print()

    # Print the smoothed learning curve for reproduction in the post.
    print("Smoothed learning curve (mean return per 100 episodes):")
    for ep, ret in smoothed:
        print(f"  episode {ep:5d}: {ret:+.4f}")

    return policy, history, smoothed, pre, post


if __name__ == "__main__":
    main()
