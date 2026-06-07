"""Construct notebooks/ch08-rl.ipynb via nbformat.

Run this, then execute the notebook with nbconvert so every cell carries a
stored output. Kept in the repo so the notebook is reproducible from a
structured source rather than hand-edited JSON. Mirrors the Chapter 4 to 6
builders (_build_ch04_nb.py ... _build_ch06_nb.py): same conventions.

THE DEFINING CONSTRAINT FOR THIS CHAPTER: the notebook is NumPy-only. It
does NOT import torch. The gridworld REINFORCE run is cheap (about a minute,
seeded), so unlike Chapter 6's scaling sweep and Chapter 7's M=32 checkpoint
there is no recorded-data fallback: everything here is a genuine live re-run
of examples/reinforce_gridworld.py.

Re-run here (NumPy, from examples/reinforce_gridworld.py):
  - Before-training evaluation under the untrained policy (seed-fixed):
    mean return -0.1258, mean length 24.20 steps, success rate 11.5%.
  - 2000-episode REINFORCE training (gamma 0.99, lr 0.05, seed 0), with the
    smoothed learning curve (mean return per 100 episodes) recorded.
  - After-training evaluation: mean return +0.9278, 8.21 steps, 100% success.
  - The trained greedy trajectory (the Manhattan path to the goal).

Figure generated: ../book/figures/ch08-gridworld-learning-curve.pdf
  (mean return vs episode, smoothed).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(text):
    cells.append(new_code_cell(text))


# ---------------------------------------------------------------------------
md(r"""# Chapter 8: Reinforcement Learning

Paired notebook for Chapter 8 of *Inductive Biases in Neural Networks*, the
book's closing chapter. It runs the one experiment the chapter rests on: a
REINFORCE policy-gradient agent learning to cross a $5 \times 5$ gridworld,
implemented in pure NumPy in `examples/reinforce_gridworld.py`.

**This notebook is NumPy-only. It does not import `torch`.** The gridworld
run is cheap (roughly a minute, fully seeded), so every number the chapter
prose quotes is regenerated here live, not cited from a recorded table.

**Re-run here (NumPy):**
1. The untrained policy's before-training behaviour (mean return, episode
   length, success rate).
2. A 2000-episode REINFORCE run ($\gamma = 0.99$, learning rate $0.05$, a
   moving-average return baseline), recording the smoothed learning curve.
3. The trained policy's after-training behaviour and its greedy trajectory.

The policy network, `PolicyMLP`, is a one-hidden-layer MLP with a $\tanh$
hidden layer and a softmax over four actions: deliberately the same
`Linear` $\to$ `Tanh` $\to$ `Linear` $\to$ `SoftmaxCrossEntropy` stack the
`scratchnn` chapters built, just vectorized in NumPy so a few times $10^4$
forward passes run in seconds. The point of the chapter is that its update
rule, the softmax cross-entropy gradient (predicted distribution minus the
one-hot of the chosen action) scaled by the return, is the supervised
gradient already derived in Chapter 1.

**Seeds:** the policy is built with `seed=0`; training uses `seed=0` for the
rollout sampler; the before/after evaluations use `seed=999`; the sampled
pre-training trajectory uses `seed=42`. All fixed, so the run is
reproducible.""")

# ---------------------------------------------------------------------------
md(r"""## Setup

Everything comes from `examples/reinforce_gridworld.py`: the environment
(`step`, `one_hot`, `state_index`), the policy (`PolicyMLP`), the REINFORCE
loop (`rollout`, `returns_to_go`, `train`), and the evaluators (`evaluate`,
`greedy_trajectory`, `sample_trajectory`). `import numpy` is the only
heavyweight import; there is deliberately no `import torch` anywhere.""")

code(r"""import os
import sys

import numpy as np

# The NumPy gridworld and the REINFORCE loop live in the examples/ directory
# of the parent scratchnn repo.
EXAMPLES = "/home/spinoza/github/repos/scratchnn/examples"
sys.path.insert(0, EXAMPLES)

import reinforce_gridworld as gw
from reinforce_gridworld import (
    PolicyMLP, rollout, returns_to_go, train,
    evaluate, greedy_trajectory, sample_trajectory,
    GRID, N_STATES, N_ACTIONS, START, GOAL,
    STEP_REWARD, GOAL_REWARD, MAX_STEPS, GAMMA,
)

# Guard the no-torch invariant: this notebook must never import torch.
assert "torch" not in sys.modules, "this notebook is NumPy-only; torch must not be imported"

print("numpy", np.__version__)
print("no torch imported:", "torch" not in sys.modules)
print(f"grid {GRID}x{GRID}  start {START}  goal {GOAL}")
print(f"reward {STEP_REWARD}/step + {GOAL_REWARD} at goal  "
      f"gamma {GAMMA}  max steps {MAX_STEPS}")""")

# ---------------------------------------------------------------------------
md(r"""## The environment and the optimal return

The agent starts at $(0, 0)$ and must reach the goal $(4, 4)$. Each step
costs $-0.01$; entering the goal pays $+1.0$ and ends the episode, which
otherwise ends after 25 steps. The shortest path is the Manhattan distance,
$4 + 4 = 8$ steps, so the optimal episode pays seven step-penalties before
the goal-reward step:

$$R^\star = 1.0 - 0.01 \times 7 = 0.93.$$

That $+0.93$ is the ceiling the learning curve should approach.""")

code(r"""manhattan = (GOAL[0] - START[0]) + (GOAL[1] - START[1])
optimal_return = GOAL_REWARD + STEP_REWARD * (manhattan - 1)
print(f"Manhattan distance start->goal : {manhattan} steps")
print(f"optimal return  1.0 - 0.01*{manhattan-1} = {optimal_return:+.2f}")""")

# ---------------------------------------------------------------------------
md(r"""## Before training: the untrained policy wanders

With the policy network freshly initialized (`seed=0`), the action
distribution at every state is close to uniform over the four moves. We
evaluate it over 200 sampled trajectories (`seed=999`). The expected return
is negative: most episodes pay the per-step cost and hit the 25-step limit
without ever stumbling into the goal.""")

code(r"""init_policy = PolicyMLP(seed=0)
pre = evaluate(init_policy)
print("Before training (untrained policy):")
print(f"  mean return  {pre['mean_return']:+.4f}")
print(f"  mean length  {pre['mean_length']:5.2f} steps")
print(f"  success rate {pre['success_rate']:.1%}")""")

code(r"""# A single sampled trajectory under the untrained policy (seed=42): it
# bounces around the grid and runs out the 25-step budget.
rng_demo = np.random.default_rng(42)
cells_pre, total_pre = sample_trajectory(init_policy, rng_demo)
print(f"sample pre-training trajectory: {len(cells_pre)-1} steps, "
      f"return {total_pre:+.4f}")
print(f"  reaches goal: {cells_pre[-1] == GOAL}")
print(f"  path: {cells_pre}")""")

# ---------------------------------------------------------------------------
md(r"""## Training: 2000 episodes of REINFORCE

The training loop (`train`, reproduced from the source file) is one screen
of code. For each episode it rolls out under the current policy, computes the
returns-to-go $G_t$ by a single backward pass over the rewards, subtracts a
moving-average baseline to get the per-step weights $w_t = G_t - b$, and
takes one SGD step on $-\sum_t \log \pi_\theta(a_t \mid s_t)\, w_t$. The
per-step gradient of that objective is exactly the softmax cross-entropy
gradient, the predicted distribution minus the one-hot chosen action, scaled
by $w_t$: this is the `grads_for_episode` method, whose body is

```
dlogits = P.copy()                 # P = softmax(logits), shape (T, 4)
dlogits[arange(T), actions] -= 1.0 # p - onehot(a) at each step
dlogits *= weights[:, None]        # scale each step by w_t = G_t - baseline
```

which is the chapter's bridge in code. Training prints the mean return over
the last 100 episodes every 100 episodes; we keep that smoothed curve for the
figure.""")

code(r"""policy, history, smoothed = train(n_episodes=2000, lr=0.05, seed=0,
                                  log_every=100)
print()
print(f"trained on {len(history)} episodes; "
      f"{len(smoothed)} smoothed checkpoints recorded")""")

# ---------------------------------------------------------------------------
md(r"""## After training: a near-optimal policy

The same 200-trajectory evaluation (`seed=999`), now under the trained
policy. Every sampled trajectory reaches the goal, and the mean length is
barely above the optimal 8 steps.""")

code(r"""post = evaluate(policy)
print("After 2000 episodes of REINFORCE:")
print(f"  mean return  {post['mean_return']:+.4f}")
print(f"  mean length  {post['mean_length']:5.2f} steps")
print(f"  success rate {post['success_rate']:.1%}")""")

code(r"""# The greedy (argmax-action) trajectory: the policy walks a shortest path.
cells_post, total_post = greedy_trajectory(policy)
arrow_path = " -> ".join(str(c) for c in cells_post)
print(f"greedy trajectory: {len(cells_post)-1} steps, return {total_post:+.4f}")
print(f"  reaches goal: {cells_post[-1] == GOAL}")
print(f"  optimal length: {cells_post[-1] == GOAL and (len(cells_post)-1) == manhattan}")
print(f"  path: {arrow_path}")""")

# ---------------------------------------------------------------------------
md(r"""## The smoothed learning curve

Mean return per 100 episodes across the 2000-episode run. The shape is the
one every policy-gradient run on a sparse-reward task has: a fast climb in
the first roughly 200 episodes (the first chance trajectory that reaches the
goal puts strong positive weight on the actions that got there), then a slow
refinement as the policy concentrates on the shortest paths.""")

code(r"""print("episode   mean return (smoothed over 100)")
for ep, ret in smoothed:
    print(f"  {ep:5d}   {ret:+.4f}")""")

# ---------------------------------------------------------------------------
md(r"""### Figure: the learning curve

Mean return against episode, with the optimal return ($+0.93$) and the
untrained baseline drawn for reference. Saved to
`../book/figures/ch08-gridworld-learning-curve.pdf` and placed in the
chapter's worked-example section.""")

code(r"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "../book/figures/ch08-gridworld-learning-curve.pdf"
eps = [e for e, _ in smoothed]
rets = [r for _, r in smoothed]

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(eps, rets, marker="o", color="#4C72B0", label="mean return (per 100 ep)")
ax.axhline(optimal_return, linestyle="--", color="#55A868",
           label=f"optimal return (+{optimal_return:.2f})")
ax.axhline(pre["mean_return"], linestyle=":", color="#C44E52",
           label=f"untrained ({pre['mean_return']:+.3f})")
ax.set_xlabel("episode")
ax.set_ylabel("mean return")
ax.set_title(r"REINFORCE on the $5\times5$ gridworld (2000 episodes)")
ax.set_ylim(min(pre["mean_return"], min(rets)) - 0.1, optimal_return + 0.1)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG)
print(f"saved {FIG}")""")

# ---------------------------------------------------------------------------
md(r"""## Results

Everything the chapter prose quotes, in one place. Every number below is a
live NumPy re-run of `examples/reinforce_gridworld.py` (seeds as documented
at the top); there is no cited or recorded fallback for this chapter.""")

code(r"""print("=" * 60)
print("RE-RUN (NumPy, this notebook) -- REINFORCE gridworld")
print("=" * 60)
print(f"grid {GRID}x{GRID}, start {START}, goal {GOAL}, "
      f"reward {STEP_REWARD}/step + {GOAL_REWARD} at goal")
print(f"gamma {GAMMA}, max steps {MAX_STEPS}, lr 0.05, 2000 episodes, seed 0")
print()
print(f"optimal return (8-step Manhattan path) : {optimal_return:+.2f}")
print()
print("Before training (untrained policy):")
print(f"  mean return  {pre['mean_return']:+.4f}")
print(f"  mean length  {pre['mean_length']:5.2f} steps")
print(f"  success rate {pre['success_rate']:.1%}")
print()
print("After 2000 episodes:")
print(f"  mean return  {post['mean_return']:+.4f}")
print(f"  mean length  {post['mean_length']:5.2f} steps")
print(f"  success rate {post['success_rate']:.1%}")
print()
print("Greedy trajectory (argmax actions):")
print(f"  {len(cells_post)-1} steps, return {total_post:+.4f}")
print(f"  {' -> '.join(str(c) for c in cells_post)}")
print()
print("Smoothed learning curve (mean return per 100 episodes):")
for ep, ret in smoothed:
    print(f"  episode {ep:5d}: {ret:+.4f}")
print()
print("Figure written:")
print(f"  {FIG}")""")


nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}
with open("ch08-rl.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote ch08-rl.ipynb")
