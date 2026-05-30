# RL Intro: From Per-Example Labels to Per-Trajectory Returns (Outline)

## Title and thesis

**Working title:** "Reinforcement Learning: When the Signal is a Scalar Reward Over Trajectories"

**One-line thesis:** RL is the paradigm where the training signal is a scalar reward summed along a trajectory, not a label at every step. The distinctive hard part is *temporal credit assignment*, not the absence of fully-supervised labels.

**Series position:** Post 7 of 7, the closing post. The first six posts walked the supervised arc, varying the architecture (MLP, CNN, RNN, Transformer) and the output head (sigmoid, softmax, mixture, factored) along two parallel axes of inductive bias. This post is the missing third leg: the *paradigm* axis. It introduces RL not by trying to do it justice (a dedicated series will) but by drawing the line between supervised and RL precisely enough that the reader can see what changes when the signal changes.

## Pedagogical arc

1. Frame the trichotomy: supervised / unsupervised / RL. Locate self-supervised learning inside supervised, mathematically.
2. Show that the supervised-to-RL reduction (reward = $-\mathrm{NLL}$) exists but discards what makes RL hard.
3. Name the hard problem: credit assignment over time.
4. Set up MDPs as the standard formalism.
5. Derive REINFORCE from first principles. Read it back as "supervised cross-entropy, weighted by return."
6. Train a tiny policy network on a tiny task. Numpy, not pure Python (justify).
7. Point at AIXI as the theoretical north star, parallel to Solomonoff induction in the LM post.
8. Re-cast the supervised series' inductive-bias framework on the RL axis: reward shaping, policy architecture, exploration, algorithm class.
9. Close the supervised series and hand off to a dedicated RL series.

## Section-by-section breakdown

### Section 1. The trichotomy

- **Supervised.** Given $(x, y)$ pairs, learn $f_\theta$ such that $f_\theta(x) \approx y$. One gradient per example. The loss tells you the answer was wrong *and how to be less wrong on this exact input*.
- **Unsupervised.** Given $x$ alone, learn structure: density, clusters, manifolds, factors. No target.
- **Reinforcement learning.** Given an environment and a scalar reward, learn a policy $\pi_\theta(a \mid s)$ that maximizes expected return. The signal is a number per trajectory, not a label per example.
- **Where SSL fits.** Self-supervised learning is supervised with auto-generated labels (the next token, the masked patch, the contrastive partner). Mathematically identical to supervised. Practically distinct because the data supply is unbounded, which changes scaling. SSL is not a fourth paradigm; it is supervised learning that solved its label problem.
- The boundary that matters is not "do we have labels" but "do we have *per-example* gradient signal." Supervised and SSL do. RL does not.

### Section 2. The "you can torture supervised into RL" non-equivalence

- A common move: "supervised is just RL with reward $r = -\mathrm{NLL}(y \mid x)$." Technically true.
- What it loses: the reward arrives at the same step as the action, and the gradient of the reward with respect to the action is directly available. The hard parts of RL (delayed reward, sparse reward, exploration, off-policy correction) all collapse to nothing.
- The genuine difficulty of RL is *credit assignment over time*. A trajectory $(s_0, a_0, s_1, a_1, \ldots, s_T)$ produces one scalar return $R = \sum_t r_t$. You took, say, 200 actions. Which ones were responsible for the return being high or low?
- REINFORCE, advantage estimation, TD learning, GAE, eligibility traces, hindsight relabeling, off-policy corrections: every named technique in RL is a piece of credit-assignment machinery. Supervised learning never needs that machinery because it never has the credit-assignment problem.
- The reframing matters because it lets a reader who has only ever done supervised learning see *why* RL is hard, not just *that* it is.

### Section 3. The MDP framing

- **Agent** and **environment** in a loop. At step $t$ the agent observes state $s_t$, takes action $a_t \sim \pi(\cdot \mid s_t)$, the environment returns reward $r_{t+1}$ and next state $s_{t+1}$.
- **Markov property.** The transition distribution $p(s', r \mid s, a)$ depends only on the current $(s, a)$, not on history. The state is whatever you need to remember.
- **Discount.** $\gamma \in [0, 1)$ weights future rewards. The discounted return from step $t$ is
  $$G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}.$$
- **Objective.** Maximize the expected return $J(\pi) = \mathbb{E}_{\tau \sim \pi}[R(\tau)]$, where $\tau = (s_0, a_0, r_1, s_1, \ldots)$ is a trajectory sampled by following $\pi$.
- **Value functions.** $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ and $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$. These are the obvious quantities to estimate.
- Read in words: a policy is a conditional distribution over actions given states; a return is the sum of (discounted) rewards along the trajectory the policy induces; we want the policy whose induced trajectories have the highest expected return.

### Section 4. Why credit assignment is the hard problem

- Supervised: you observe a loss for every example. You know which input caused which loss. The gradient pins the blame precisely.
- RL: you observe a return at the end (or sparse rewards along the way). You took many actions in between. *Which action mattered?*
- Three families of answers, each a different bet about how to split the credit:
  - **Monte Carlo.** Wait until the episode ends; attribute the *whole* return to *every* action in the trajectory (perhaps with a discount). High variance, unbiased.
  - **Temporal difference (TD).** Don't wait. Use a learned value estimate $V(s_{t+1})$ as a bootstrap target for $V(s_t)$. Lower variance, biased by the value estimate.
  - **Policy gradient.** Skip the credit-assignment question for a single trajectory; estimate the gradient by averaging the score-function expression (Section 5) over many trajectories. The averaging is what makes it work.
- Actor-critic, GAE, n-step returns, eligibility traces are interpolations between these three.

### Section 5. REINFORCE, derived

- The simplest policy-gradient algorithm. Worth deriving in full because the derivation is short, the result is iconic, and the bridge to supervised learning is exact.
- Parameterize the policy: $\pi_\theta(a \mid s)$. Think "neural network with a softmax head over discrete actions," or "neural network outputting the parameters of a continuous distribution."
- Objective:
  $$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau p_\theta(\tau) R(\tau),$$
  where $p_\theta(\tau)$ is the probability of trajectory $\tau$ under policy $\pi_\theta$.
- **The log-derivative trick (likelihood-ratio identity).** For any distribution $p_\theta$,
  $$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau).$$
  This is just $\nabla \log x = \nabla x / x$ rearranged.
- Apply it:
  $$\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta p_\theta(\tau) \cdot R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log p_\theta(\tau) \cdot R(\tau)].$$
- Now factor the trajectory probability. The environment's dynamics $p(s' \mid s, a)$ do *not* depend on $\theta$, so they drop out of the gradient. Only the policy factors $\pi_\theta(a_t \mid s_t)$ remain:
  $$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t).$$
- The classical REINFORCE estimator (reward-to-go form):
  $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right].$$
  Read in words: the gradient of the expected return is, on average across sampled trajectories, the sum over time steps of the log-likelihood gradient of the action taken, weighted by the return earned from that step onward.
- **Variance reduction via baselines.** Subtracting any state-dependent baseline $b(s_t)$ from $G_t$ leaves the expectation unchanged but reduces variance:
  $$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t))\right].$$
  A common baseline is a learned value estimate $V_\phi(s_t)$. Then $G_t - V_\phi(s_t)$ is an estimate of the advantage $A^\pi(s, a)$. This is the half-step from REINFORCE to actor-critic.

### Section 6. The bridge: REINFORCE is weighted supervised cross-entropy

- Look at the per-step term in REINFORCE: $-\nabla_\theta \log \pi_\theta(a_t \mid s_t)$.
- That is *exactly* the supervised cross-entropy gradient on the pair $(s_t, a_t)$, treating $a_t$ as a one-hot label and $\pi_\theta(\cdot \mid s_t)$ as the predicted class distribution. The same expression we derived in the scratchnn walkthrough for `SoftmaxCrossEntropy`.
- Policy gradient is therefore supervised cross-entropy on the actions the agent took, with each sample weighted by $G_t$ (or by an advantage).
- **Behavioral cloning** is the special case where the weight is constant (or simply absent): given a dataset of expert $(s, a)$ pairs, fit $\pi_\theta$ by supervised cross-entropy. Pure supervised learning of a policy.
- Conceptually: in supervised learning the dataset tells you what the right action was. In RL the *return* tells you, after the fact, *how good* the action you took was, and you weight your gradient accordingly. The loss surgery is the same; the labels and weights come from different places.
- This is the natural conceptual entry point for a reader who knows scratchnn. The math you already know carries over; what changes is who provides the labels (you vs. the environment) and how the per-sample importance is scored (uniformly vs. by return).

### Section 7. Worked example: tiny gridworld policy gradient

- **Why gridworld over CartPole or bandit?**
  - A multi-armed bandit has no state, so the credit-assignment story shrinks to a single step and the post's main point is hidden.
  - CartPole has continuous state and a fairly dense reward. The reward density makes credit assignment look easy.
  - A small 2D gridworld (say, $5 \times 5$, agent navigates from a corner to a goal) has a small discrete state space, *sparse* reward (one unit at the goal, zero elsewhere), and an obvious visualization (the value function as a heatmap over the grid). Credit assignment is on the page; you can *see* the value-of-progress propagate backward from the goal as training proceeds.
- **Setup.**
  - State: agent position $(r, c) \in \{0, \ldots, 4\}^2$. One-hot encoded into a 25-dim vector.
  - Actions: up, down, left, right (four discrete actions).
  - Reward: $+1$ on entering the goal cell, $0$ otherwise. Episode ends at the goal or after 50 steps.
  - Discount $\gamma = 0.99$.
- **Policy network.** A small MLP: 25-input one-hot, one hidden layer of width 32 with $\tanh$, then 4 output logits. The output head is `SoftmaxCrossEntropy`-shaped: logits in, log-probabilities used in the loss. The reader recognizes this exactly from scratchnn.
- **Training loop.** REINFORCE with a value baseline.
  1. Roll out an episode under $\pi_\theta$.
  2. Compute returns-to-go $G_t$ for every step.
  3. Compute advantages $A_t = G_t - V_\phi(s_t)$.
  4. Policy update: gradient ascent on $\sum_t \log \pi_\theta(a_t \mid s_t) \cdot A_t$.
  5. Value update: regress $V_\phi(s_t)$ against $G_t$ (MSE).
- **Implementation choice: NumPy, not pure Python.** Justify in-line:
  - A typical run needs $10^4$ episodes of up to 50 steps, roughly $5 \cdot 10^5$ forward passes. Each forward pass through even a small MLP is dominated by Python-loop overhead in the pure-Python style. The cost is 100 to 1000x what NumPy would do.
  - More importantly, the *interesting* part is no longer the layer implementation; it is the training loop and the gradient estimator. The MLP itself is the same MLP from scratchnn. NumPy lets the post be about RL, not about list comprehensions.
  - This is the same honest acknowledgement the Transformer post makes when it switches to PyTorch. The pedagogy of "every parameter is a list of floats" did its job in posts 1 to 5; this post is about the loss, not the layer.
- **What to show.** Training curve (mean return per episode vs. episode index), pre- and post-training trajectories on the grid (random walk vs. straight-line policy), and the learned value function as a heatmap.

### Section 8. AIXI as theoretical forward-pointer

- Solomonoff induction (introduced in the language-model post) is the optimal incomputable *predictor*: a Bayesian mixture over all computable environments, weighted by program length. AIXI is its action-taking sibling.
- AIXI selects the action that maximizes expected discounted reward under the Solomonoff prior over environments. Informally,
  $$a_t^* = \arg\max_{a_t} \sum_{o_t} \xi(o_t \mid h_{<t}\, a_t) \cdot V^*(h_{<t}\, a_t\, o_t),$$
  where $h_{<t}$ is the interaction history up to step $t$, $o_t$ packs the next reward and observation, $\xi$ is the Solomonoff prior over environments (programs that could be generating the observations), and $V^*$ is the optimal value function under that prior.
- Decoded: AIXI maintains a posterior over all possible environments consistent with what it has seen, and at every step it takes the action whose *expected* future return, averaged over that posterior, is highest.
- AIXI is literally Solomonoff induction (post 4) plus Bayesian decision theory plus reward maximization, fastened together. The dedicated RL series, forthcoming, will unpack each component in its own post.
- **The parallel.** The supervised arc had Solomonoff induction as the unachievable but illuminating prediction ideal; modern LMs are practical, computable approximations. RL has AIXI as the unachievable but illuminating action ideal; modern RL algorithms (DQN, PPO, MuZero, decision transformers) are practical, computable approximations.
- The pattern that closes the series: in both arcs, the architecture is the computable inductive bias substituted for an incomputable optimum. You can't run Solomonoff. You can run a Transformer. You can't run AIXI. You can run PPO. The architecture *is* where you compromise.

### Section 9. Inductive bias on the RL axis

The supervised series argued that architecture (post 3 to 6) and output head (post 2) are parallel axes of inductive bias. RL extends the picture with more design surface, each piece of which encodes assumptions about the task structure.

1. **Reward shaping.** The reward function is itself a prior about the value landscape. Shaping rewards (a small positive reward for getting closer to the goal, for example) injects information about which intermediate states are progress. Sparse rewards are minimal-prior; dense shaping is heavy-prior. The shaping can also bias the policy in the wrong direction if your hint is wrong, which has its own literature (reward hacking, Goodhart's law).
2. **Policy architecture.** Whether $\pi_\theta$ is an MLP, a CNN over pixels, an RNN over trajectories, or a Transformer over state-action histories is the *same* architectural choice the supervised series catalogued. The same inductive bias applies, just in service of action selection rather than prediction.
3. **Exploration strategy.** Epsilon-greedy, entropy bonus, intrinsic curiosity, UCB, Thompson sampling: each is a prior about *where* useful information is. Epsilon-greedy assumes "uniform random is good enough sometimes." Curiosity assumes "novel states are valuable." Thompson sampling assumes a posterior over models and exploits its variance. Different priors, different exploration behavior.
4. **Algorithm class.** On-policy vs off-policy, model-free vs model-based, value vs policy vs actor-critic: each commits to a different structural assumption about what is cheap, what is reliable, and what generalizes. Off-policy methods assume past trajectories from different policies still inform the current policy; model-based methods assume a learned dynamics model will pay off the cost of fitting it.

All of these are inductive biases in exactly the same sense that "translation equivariance" is for a CNN. The pattern from posts 2 to 6 generalizes to RL, with more axes.

### Section 10. Closing the supervised series

- This is the end of the seven-post arc. The thread was: supervised learning + a parametric model + a loss = a paradigm with two main axes of inductive bias (architecture and output head). The seven posts laid out concrete instances along both axes.
- RL is what the rest of machine learning looks like when the per-example label disappears. It deserves its own arc.
- Modern systems blur the boundary. RLHF (reinforcement learning from human feedback) fine-tunes language models with a reward model that itself was supervised. AlphaZero uses supervised distillation against MCTS targets. Decision transformers cast offline RL as supervised sequence modeling on (return, state, action) tuples. The boundary is conceptual, not categorical, and a serious treatment has to thread back and forth across it.
- Pointer to the dedicated RL series (forthcoming), in which AIXI is the theoretical north star, the way Solomonoff induction framed the language-model post.

## Figures

1. **Agent-environment loop.** The canonical diagram: agent emits $a_t$, environment emits $(s_{t+1}, r_{t+1})$, arrow back to the agent. Captioned with the variable names from Section 3.
2. **Learning curve.** Mean episode return vs. episode index for the gridworld run. Expect a noisy climb from near-zero toward the maximum possible (one, for the sparse setup). One panel suffices; smoothing visible.
3. **Trajectories: before and after.** A grid drawn twice. Left panel: trajectories under the untrained random policy (a tangle of self-intersecting walks, mostly hitting the step limit). Right panel: trajectories under the trained policy (mostly straight or nearly-straight lines to the goal). The visual punch of the worked example.
4. **Value function heatmap.** $V_\phi(s)$ over the grid, drawn as a colored cell at each $(r, c)$. The value should radiate out from the goal cell with a $\gamma$-induced exponential falloff. Annotated to call this out. This is the figure that makes "credit propagation over time" visible.
5. **Optional: REINFORCE-as-weighted-cross-entropy schematic.** Two columns side by side. Left: supervised cross-entropy on an $(x, y)$ pair, gradient is the standard $p - y$. Right: REINFORCE on an $(s, a)$ pair, gradient is $(p - \mathrm{onehot}(a)) \cdot G_t$. Same expression, scaled by return. This figure carries the Section 6 bridge visually.

## Math content checklist

- Trajectory probability factorization and the dropout of environment dynamics from the gradient.
- The log-derivative trick, in one line, with a sentence reading it back.
- Reward-to-go form of REINFORCE, derived from the full-return form by observing that future actions cannot affect past rewards (causality argument).
- Variance reduction by baseline, with the "expectation unchanged" argument (a baseline times the score function has zero expectation).
- Discounted return and the geometric series intuition for $\gamma$.
- AIXI's defining expression, with a one-paragraph informal decoding. Not derived; signposted as a teaser.

## Code approach

- NumPy for the gridworld example. Honest acknowledgement up front: pure Python ran the supervised series; rollouts make it untenable here, and the post is about the loss, not the layer.
- One MLP class, one rollout function, one REINFORCE update, one value-network update. Roughly 150 lines.
- The MLP itself is *the same MLP* as scratchnn, just vectorized. The reader who learned the per-layer backward pass in post 1 should be able to read the NumPy version and recognize every step.
- We do *not* pull in `gymnasium` or `stable-baselines3`. The gridworld is hand-written in maybe 30 lines. Reader sees every transition.

## Notes on tone and conformance

- No em-dashes; use commas, colons, periods, or parentheses.
- Math integrated with prose; every display equation followed by a sentence reading it back in words.
- Direct, terse, math-forward; no marketing.
- Whenever a piece of RL machinery is named (TD, GAE, actor-critic, off-policy correction), state in one sentence what credit-assignment problem it solves. Don't list techniques without function.
- Refer back to scratchnn's `SoftmaxCrossEntropy` whenever the policy-network output appears; the bridge to supervised is the post's most important conceptual lever.
- AIXI gets one section; resist the urge to develop it further. It is a pointer, not a topic, here.

## Open questions for the writer

- Whether to derive the policy-gradient theorem in the full form (sum over the stationary distribution of the policy) or stick with the trajectory form. The trajectory form is shorter and more self-contained; the stationary form connects to value iteration. The trajectory form is the right pedagogical choice for this post; the dedicated series can do the stationary form.
- Whether to include a one-paragraph note on Q-learning as a contrast to policy gradient. Pro: it gives a flavor of the value-based family. Con: it doubles the new-vocabulary load. Lean toward a one-sentence mention with a forward pointer.
- Whether the gridworld should be deterministic or stochastic transitions. Deterministic is cleaner pedagogically; stochastic makes the expectation in $J(\theta)$ less of a fiction. Lean deterministic for the visualization, mention stochastic as a one-liner extension.
