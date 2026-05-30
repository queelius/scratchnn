# Output Heads Outline: The Link Function as Inductive Bias (Post 2 of 7)

## Title and thesis

**Title:** Output Heads as Inductive Bias: Every Network is a Maximum-Likelihood Estimator

**One-line thesis:** Every supervised network is a maximum-likelihood estimator under an assumed output distribution; the link function and the matching loss together specify that likelihood, and the assumed likelihood is itself an inductive bias.

## Pedagogical arc

The reader should leave with these, in order:

1. The walkthrough trained three losses (MSE, BCE, cross-entropy). They were not three unrelated formulas. Each is the negative log-likelihood of a particular assumed distribution for $y \mid x$.
2. The link function (identity, sigmoid, softmax) turns the network's raw output $z$ into the right kind of parameter (a mean, a probability, a class distribution) for that distribution.
3. The pairing is not arbitrary. With the **canonical link** of an exponential-family distribution, the gradient $\partial \mathrm{NLL} / \partial z$ collapses to $\hat{p} - y$. The walkthrough's "both losses had $\mathbf{p} - \mathbf{y}$" observation is one instance of a general theorem.
4. Picking the right link is an inductive bias about the data-generating process for $y$. Count data wants log + Poisson; bounded data wants logit + Beta; heteroscedastic data wants a two-output Gaussian head.
5. **Punchline:** Architecture (post 3 onward) and output head (this post) are *parallel* axes of inductive bias. Both buy sample efficiency. Both cost you accuracy when mismatched to the problem.

## Section-by-section breakdown

### Section 1: The unifying frame

- The walkthrough trained three losses. State them as one thing:

$$L(\theta) = -\frac{1}{N} \sum_{n=1}^{N} \log p_\theta(y_n \mid x_n).$$

  Minimizing the loss *is* maximum likelihood under an assumed conditional distribution $p_\theta(y \mid x)$. The network parameterizes that distribution; gradient descent moves the parameters toward higher likelihood of the observed data.
- The output head is the bridge. The network produces a raw vector $z = f_\theta(x)$. A link function $g^{-1}$ turns $z$ into the natural parameter (or mean) of the assumed distribution. The loss is the NLL of $y$ under that distribution.
- That is the entire story. The rest of the post unpacks it.
- Reuse the organizing principle from `scratchnn`: **the network produces logits; the loss interprets them**. The link function and the assumed distribution *are* the interpretation.

### Section 2: The catalogue

A table the reader can return to. Five canonical pairings, each one a member of the exponential family with its canonical link:

| Output type        | Link              | Loss            | Assumed distribution     |
|--------------------|-------------------|-----------------|--------------------------|
| Real-valued        | identity          | MSE             | Gaussian (fixed variance)|
| Binary             | logit (sigmoid)   | BCE             | Bernoulli                |
| Categorical        | softmax           | cross-entropy   | Categorical              |
| Count              | log (exp)         | Poisson NLL     | Poisson                  |
| Positive continuous| reciprocal        | Gamma NLL       | Gamma                    |

- The first three are what the walkthrough already built. The rest of the post extends the table.
- Each row is a different commitment about what kind of object $y$ is. Real-valued and unbounded? Gaussian. A 0 or 1? Bernoulli. One of $K$ mutually exclusive classes? Categorical. A non-negative integer (event counts)? Poisson.
- The link makes the network's unconstrained $z \in \mathbb{R}$ a valid parameter: a real mean for Gaussian, a probability in $[0, 1]$ for Bernoulli, a probability vector on the simplex for Categorical, a positive rate for Poisson.

### Section 3: The canonical link theorem

- The walkthrough observed that BCE and softmax cross-entropy both have $\partial L / \partial z = \mathbf{p} - \mathbf{y}$. Not a coincidence. Here is the general statement.
- An exponential family in natural form:

$$p(y \mid \eta) = h(y) \exp\big(\eta \cdot T(y) - A(\eta)\big),$$

  where $\eta$ is the **natural parameter**, $T(y)$ is the sufficient statistic, and $A(\eta)$ is the log-partition function. For Bernoulli, $\eta = \mathrm{logit}(p)$; for Categorical, $\eta_i = \log p_i$ up to a shift; for Gaussian (fixed variance), $\eta = \mu / \sigma^2$; for Poisson, $\eta = \log \lambda$.
- A foundational identity: $A'(\eta) = \mathbb{E}[T(y)] = \hat{p}$, the expected sufficient statistic. (Differentiate $\int h(y) \exp(\eta T(y) - A(\eta)) \, dy = 1$ in $\eta$.)
- Now suppose the network's output $z$ *is* the natural parameter, $\eta = z$. This is the **canonical link** choice. The NLL is

$$-\log p(y \mid z) = -\log h(y) - z \cdot T(y) + A(z),$$

  and so

$$\frac{\partial}{\partial z} \big[ -\log p(y \mid z) \big] = -T(y) + A'(z) = \hat{p} - y$$

  (reading $y$ as $T(y)$, the sufficient statistic; for Categorical, that means the one-hot).
- This is the theorem. **For any exponential-family distribution with its canonical link, the gradient of the NLL with respect to the network's output is the expected sufficient statistic minus the observed.**
- Consequence: the BCE-and-softmax pattern from the walkthrough is not two parallel coincidences. It is the same theorem twice. Poisson with log link, the next example, will do it a third time.
- Pedagogical aside: this is also why the canonical link is a *natural* choice, not just a convenient one. The gradient is geometrically a residual.

### Section 4: Worked example 1, identity + Gaussian + MSE (review)

- This is the regression baseline from walkthrough §1. Brief recap.
- Assumed distribution: $y \mid x \sim \mathcal{N}(\mu(x), \sigma^2)$ with $\sigma^2$ fixed and absorbed into the constant.
- Link: identity. The network's output $z$ *is* $\mu$.
- NLL (dropping constants): $\tfrac{1}{2}(y - z)^2$. Mean squared error, exactly.
- Gradient: $\partial L / \partial z = z - y = \hat{p} - y$. The canonical-link theorem, instance one.
- Inductive bias: errors are symmetric, additive, and Gaussian. If the true distribution is heavy-tailed or skewed, MSE pays for that with biased estimates.

### Section 5: Worked example 2, logit + Bernoulli + BCE (review)

- From walkthrough §2. Brief recap.
- Assumed distribution: $y \mid x \sim \mathrm{Bernoulli}(p(x))$, $y \in \{0, 1\}$.
- Link: logit, $\eta = \log \tfrac{p}{1 - p} = z$. Inverse link is sigmoid, $p = \sigma(z)$.
- NLL: $-y \log p - (1 - y) \log(1 - p)$, computed stably from $z$ as in the library.
- Gradient: $\partial L / \partial z = p - y$. Instance two.
- Inductive bias: the response is binary; a probabilistic prediction is well-calibrated when training converges.

### Section 6: Worked example 3, softmax + Categorical + cross-entropy (review)

- From walkthrough §3. Brief recap.
- Assumed distribution: $y \mid x \sim \mathrm{Cat}(\mathbf{p}(x))$, $y \in \{0, \ldots, K-1\}$.
- Link: softmax, $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$.
- NLL: $-\log p_y = \mathrm{logsumexp}(\mathbf{z}) - z_y$, the stable form.
- Gradient: $\partial L / \partial z_i = p_i - [i = y]$, i.e. $\mathbf{p} - \mathbf{y}$ with $\mathbf{y}$ the one-hot. Instance three.
- Inductive bias: the classes are mutually exclusive and exhaustive. The model cannot represent "two classes at once", a structural prior.

### Section 7: Worked example 4, log link + Poisson + Poisson NLL (new)

The first genuinely new pairing. This is where the catalogue starts paying off.

- Assumed distribution: $y \mid x \sim \mathrm{Poisson}(\lambda(x))$, $y \in \{0, 1, 2, \ldots\}$, non-negative integers.
- Link: log, $\eta = \log \lambda = z$, so $\lambda = e^z$. The exponentiation guarantees the rate is positive without any clamping.
- NLL: $-\log p(y \mid \lambda) = \lambda - y \log \lambda + \log y! = e^z - y z + \mathrm{const}$. The constant $\log y!$ does not depend on $z$ and drops out of the gradient.
- Gradient: $\partial L / \partial z = e^z - y = \lambda - y = \hat{p} - y$. Instance four. The canonical-link theorem again, with no new derivation.
- **Synthetic experiment.** A 1-D regression where the true rate is non-monotonic, e.g. $\lambda(x) = 2 + 5 \sin(\pi x)$ clipped at $0.1$ so it stays positive, $x \in [0, 2]$. Sample $y_n \sim \mathrm{Poisson}(\lambda(x_n))$ for $N = 500$.
- Train two networks of identical architecture (one hidden Tanh layer, say $1 \to 8 \to 1$):
  - **(a)** identity + MSE, treating it as ordinary regression.
  - **(b)** log + Poisson NLL.
- Findings to report (numbers to fill in from a run; placeholders below):
  - **(a)** fits the conditional mean in the bulk but predicts *negative* rates in the low-rate region, which is impossible for counts. Test NLL: $\sim$ (placeholder).
  - **(b)** never predicts a negative rate (cannot, by construction: $e^z > 0$). Converges in fewer epochs because the link encodes a constraint the MSE network has to discover. Test NLL: $\sim$ (placeholder, lower than (a)).
- Suggested numbers for a clean writeup: $N = 500$, $50$ epochs, $\mathrm{lr} = 0.05$, $\mathrm{batch} = 32$, seed fixed for reproducibility.
- The inductive bias is sharp and concrete: "$y$ is a non-negative integer count." The log link bakes the non-negativity in; the Poisson NLL accounts for the fact that variance grows with the mean (Poisson has $\mathrm{Var} = \mathrm{Mean}$). MSE assumes constant variance and gets neither right.

### Section 8: Worked example 5, heteroscedastic Gaussian (two-output head)

The first example where the network outputs *more than one parameter per example*. A clean break from one-output heads.

- Assumed distribution: $y \mid x \sim \mathcal{N}(\mu(x), \sigma(x)^2)$ where both $\mu$ and $\sigma$ depend on $x$.
- Head: a single `Linear` layer with $n_{\mathrm{out}} = 2$. Call the outputs $z_\mu$ and $z_s$. Set $\mu = z_\mu$ (identity link on the mean) and $\sigma = \mathrm{softplus}(z_s)$ (smooth positive link on the scale). Softplus is $\log(1 + e^{z})$, the natural smooth analogue of "positive part".
- NLL (dropping the $\log \sqrt{2 \pi}$ constant):

$$-\log p(y \mid \mu, \sigma) = \frac{(y - \mu)^2}{2 \sigma^2} + \log \sigma.$$

- Gradients, hand-derived:
  - $\partial L / \partial \mu = (\mu - y) / \sigma^2$, the MSE gradient *rescaled by the predicted variance*. A confident prediction (small $\sigma$) gets a large gradient; an unsure one gets a small one. The model self-attenuates.
  - $\partial L / \partial \sigma = -\,(y - \mu)^2 / \sigma^3 + 1 / \sigma$. Zero when $(y - \mu)^2 = \sigma^2$, exactly the maximum-likelihood point.
  - Chain through softplus: $\partial \sigma / \partial z_s = \sigma'(z_s) = 1 / (1 + e^{-z_s}) = \mathrm{sigmoid}(z_s)$. Tidy.
- **Synthetic experiment.** A 1-D function with input-dependent noise, e.g. $y = \sin(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, (x / 3)^2)$ over $x \in [0, 6]$. Sample $N = 400$. Noise is small on the left, large on the right.
- Train two networks:
  - **(a)** identity + MSE: fits the mean but predicts a single global confidence; over-confident in noisy regions, under-confident in clean ones.
  - **(b)** two-output Gaussian NLL: produces a mean *and* a per-input $\sigma$. Plot $\mu(x) \pm 2 \sigma(x)$ as a shaded band over $x$. The band widens where the noise is large; it tightens where the noise is small.
- Inductive bias: noise is Gaussian *and input-dependent*. The model can now report calibrated uncertainty, not just a point estimate. Compare: ensembles and dropout-style uncertainty are *post-hoc* approximations to what a heteroscedastic head models directly.

### Section 9: Worked example 6, sigmoid for bounded $[0, 1]$ regression (not classification)

The subtle teaching moment: the link and the likelihood are *independent choices*. The same link can pair with different losses.

- Target: a continuous proportion, $y \in [0, 1]$. Think "fraction of pixels lit", "vote share", "the relative humidity". Not a class label.
- Link: sigmoid. Output $z \in \mathbb{R}$, predicted proportion $\hat{p} = \sigma(z)$. The link's job is to enforce $\hat{p} \in (0, 1)$.
- **Two loss choices:**
  - **MSE on the sigmoid output:** $L = \tfrac{1}{2}(y - \sigma(z))^2$. Implicit assumption: errors are symmetric and Gaussian *in proportion space*. Often surprisingly fine in the bulk, but downweights data near the boundary where $\sigma'(z) \to 0$ (the sigmoid saturates and gradient flow stalls).
  - **Beta NLL:** $y \mid x \sim \mathrm{Beta}(\alpha(x), \beta(x))$ for some parameterization. One natural choice: predict the mean $\mu = \sigma(z_\mu)$ via the sigmoid and a "precision" $\nu = \mathrm{softplus}(z_\nu)$, then set $\alpha = \mu \nu$, $\beta = (1 - \mu) \nu$. This is the *mean-precision* parameterization of Beta. NLL involves a log Beta function, $\log B(\alpha, \beta)$.
- Same link (sigmoid), different likelihoods (Gaussian on the sigmoid output vs. Beta directly). The choice changes:
  - what is assumed about the noise distribution,
  - how the loss treats extreme values (BCE-flavoured Beta is much more sensitive there than Gaussian-on-sigmoid),
  - whether the head is one output or two.
- **Note for `scratchnn`:** the Beta NLL requires `lgamma` (`math.lgamma`), which is in the standard library and is fine. Implementation is left as an exercise; the pedagogical point is that you *can* pair sigmoid + Beta and the library does not need anything fundamentally new.
- This is the section that breaks the table: the link function is *not* uniquely determined by the assumed distribution. There is a canonical pairing (the one the theorem gives you), but it is one choice among many. The skill is recognising the choice.

### Section 10: Mixture density networks as the closing pointer

One paragraph. The natural extension beyond unimodal heads.

- When $p(y \mid x)$ is genuinely multi-modal (a robot arm with two valid joint configurations for the same end-effector pose; an inverse problem with multiple solutions), unimodal heads collapse. A Gaussian head will predict the mean of the modes, which may be in a region of *zero density*.
- A **Mixture Density Network (MDN)** outputs the parameters of a Gaussian mixture: $K$ means, $K$ scales, and $K$ mixture weights (the latter through a softmax head). The loss is mixture NLL,

$$-\log \sum_{k=1}^{K} \pi_k(x) \, \mathcal{N}(y \mid \mu_k(x), \sigma_k(x)^2).$$

- Visualisation to describe: standard regression collapses to the mean between two modes; the MDN captures each mode with its own component.
- Not implemented in `scratchnn`. Mentioned the way the walkthrough mentions autograd: this is the natural extension, the framework is the same (output head + matching NLL), the implementation is a straightforward composition of pieces already in the library plus an `lgamma`-free log-sum-exp.

### Section 11: Inductive bias, the parallel axis

The synthesising section. Pull the post's whole frame together.

- The walkthrough talked about *architectural* inductive bias: an MLP versus logistic regression on XOR; depth lets you represent functions a linear unit cannot. The CNN post (next) will sharpen this further: locality and translation equivariance are biases wired structurally into the weights.
- The output head is the *other* axis of inductive bias. It is a prior about the DGP for $y$. A matched head means:
  - **Sample efficiency.** Constraints encoded by the link (non-negativity, boundedness, simplex membership) are free. The network does not spend parameters on relearning them.
  - **Calibrated uncertainty.** A two-output Gaussian head reports per-input $\sigma$. A categorical head reports a probability vector. Both are usable by downstream decisions.
  - **Honesty about the data.** A Poisson head says "I am modelling counts." That is a statement to the reader as much as a setting on the optimiser.
- A mismatched head means:
  - Wasted capacity discovering constraints the link could have given for free.
  - Pathological predictions in regions of low data density (negative counts, point estimates between modes).
  - Misleading uncertainty (an MSE network has none; a poorly calibrated softmax is a known failure mode).
- **The two axes are independent and composable.** A CNN with a Poisson head models pixel-wise count data (photon imaging, e.g.). A transformer with a categorical head over a vocabulary is the dominant LM. An MLP with a heteroscedastic Gaussian head models input-dependent noise. The output head is orthogonal to the choice of body.

### Section 12: Library additions

Concrete additions to `src/scratchnn/neural_net.py`. All hand-derived backward, all fitting the existing `Loss` interface (`value`, `grad`, `probs`). All standard-library only.

- **`MSELoss`** (already present after the walkthrough restructure; included for completeness). Signature: takes a single logit, `value` is $\tfrac{1}{2}(z - y)^2$, `grad` is $[z - y]$, `probs` is $[z]$ (identity link).
- **`PoissonNLLLoss`** (new, ~10 lines). Takes a single logit $z$, treats $\lambda = e^z$.
  - `value`: $e^z - y z$ (constant $\log y!$ omitted; document that the loss is up to a $y$-only constant, which does not affect optimisation).
  - `grad`: $[e^z - y]$. Canonical-link form, matches the theorem.
  - `probs`: $[e^z]$, the predicted rate.
- **`GaussianNLLLoss`** (new, ~15 lines). Takes two logits $(z_\mu, z_s)$.
  - Forward: $\mu = z_\mu$; $\sigma = \mathrm{softplus}(z_s)$.
  - `value`: $\tfrac{(y - \mu)^2}{2 \sigma^2} + \log \sigma$ (constants dropped).
  - `grad`: returns the two-component vector $[\partial L / \partial z_\mu, \partial L / \partial z_s]$, derived in §8. Chain through softplus: $\partial \sigma / \partial z_s = \sigma(z_s)$ (the sigmoid).
  - `probs`: $[\mu, \sigma]$, both reported.
- Each new loss gets a `gradient_check` case in `tests/test_gradients.py`. The anchor network is a `Tanh` MLP (per the CLAUDE.md note on the ReLU kink). Tolerance stays at $10^{-4}$.
- **Out of scope:** no `BetaNLLLoss` in the library. It is mentioned in §9 as something the reader can implement; pulling it in would mean introducing `math.lgamma` to the core, which is fine technically but not pedagogically necessary.

### Section 13: Figures

Sketches; each is a small matplotlib panel produced by an `examples/` script in the spirit of the existing demos. Listed in the order they appear in the post.

1. **Identity + MSE on a sine fit.** Small scatter of $(x, y)$ with $y = \sin(x) + \mathcal{N}(0, 0.1)$; overlaid prediction line. Reference figure for "what a working regression looks like."
2. **Identity + MSE failing on Poisson count data.** Same axes as figure 3 below. Show the prediction line going *negative* in the low-rate region. Annotate "predicted rate < 0; impossible for counts."
3. **Log + Poisson NLL on the same data.** Predictions stay positive everywhere; the curve follows the non-monotonic ground truth $\lambda(x)$ closely. Side-by-side with figure 2 makes the inductive-bias point in one image.
4. **Heteroscedastic Gaussian fit with error bars.** Mean curve $\mu(x)$ as a solid line; $\pm 2 \sigma(x)$ as a shaded band. Band narrow on the left (small $x$, small noise), wide on the right (large $x$, large noise). Overlay raw data so the reader can see the band tracks the actual spread.
5. **Comparison: same body, different heads.** Bar chart or table with three rows (MSE, Poisson NLL, Gaussian NLL) and one column (test NLL on a held-out fold of the count dataset). Numerical proof that head choice changes generalisation, not just style.

All figures saved to `examples/data/` for the prose to reference. The viz quarantine still applies: figure-producing scripts live under `examples/`, not in `src/scratchnn/`.

### Section 14: Handoff

- This post covered the output-head axis of inductive bias. Catalogue, theorem, worked examples, library additions.
- The next post (CNN) starts the architecture axis. Locality and translation equivariance are biases wired into the body of the network; an output head is a bias wired into the interpretation of the body. The two compose.
- After CNN come fixed-context language modelling (MLP + softmax over a vocabulary, an MDN-style head at scale), RNNs (architectural prior for temporal recurrence), the transformer (attention), and RL (REINFORCE, which reuses softmax cross-entropy with a different target).
- Every later post reuses the frame from §1: pick a body, pick a head, write down the NLL, hand-derive the backward, train. The skill of recognising the choice is the skill these posts are trying to build.
