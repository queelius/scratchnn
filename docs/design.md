# Pure-Python Pedagogical Neural Network — Design

- **Date:** 2026-05-20
- **Status:** Approved (design); ready for implementation planning
- **Topic:** A small, dependency-free neural network library built for teaching, covering the progression logistic regression → multi-class (softmax) regression → multi-layer perceptron.

> **Packaging note:** This is the original design specification, written
> before the library was packaged. The prototype's flat `nn/` layout (see
> §4.1) is now the `src/scratchnn/` package — the README has the current
> layout. The design rationale below is otherwise unchanged.

## 1. Purpose

Build a neural network library in pure Python — standard library only, no NumPy, no numerical or matrix libraries — whose sole goal is pedagogy. It teaches, in one coherent arc:

1. **Logistic regression** — the fundamental unit.
2. **Multi-class (softmax) regression** — the same idea generalized to many classes.
3. **The multi-layer perceptron (MLP)** — logistic-style units stacked into layers.

The whole progression is expressed as *configurations of a single unified module*, not as separate codebases. Logistic and softmax regression are literally a one-`Linear`-layer `Network`; an MLP adds layers. "Laying logistic regression into a multi-layer network" is true in the code.

Simplicity, elegance, and good abstractions are the paramount design constraints. Where a choice trades cleverness for clarity, clarity wins.

## 2. Goals and Non-Goals

### Goals

- One compact, readable module that subsumes all three stages.
- Gradients are **hand-derived** and computed by an explicit **per-layer backward pass**. The chain rule is visible and local.
- A **numerical gradient checker** that verifies every analytical gradient.
- A written **walkthrough** carrying the math and the narrative, plus runnable **demos**.
- Numerical stability handled correctly (softmax, cross-entropy, sigmoid, BCE).

### Non-Goals

- No NumPy / SciPy / matrix libraries and no pytest dependency. The core library, the demos, and the tests use the standard library only. The optional visualization module (`nn/visualize.py`) is the single exception: it may use matplotlib, plus Pillow for the headless GIF fallback.
- No autograd engine. Autograd is described in a closing note of the walkthrough as the natural generalization of per-layer backward, but it is **not built**.
- No `Optimizer` abstraction, no momentum/Adam, no regularization, no learning-rate schedules. Plain mini-batch SGD only.
- No model serialization (save/load), no GPU, no batching speedups.

These exclusions are deliberate. They keep the reader's attention on the network and on backpropagation.

## 3. The Organizing Principle

**The network computes logits; the loss interprets them.**

- A `Network` is the model `f_θ`. It maps an input vector to **logits** — raw, unnormalized scores. It never computes a probability.
- A `Loss` turns logits and a target into (a) a scalar loss and (b) a gradient with respect to the logits. Interpreting logits *as* probabilities — applying sigmoid or softmax — belongs to this interpretation step.

Consequences:

- The output activation (sigmoid / softmax) is **not** a layer of the model. It lives inside the `Loss` (and is what `predict` applies). This is not a convenience hack; the model's natural output genuinely is unnormalized log-probabilities, and the loss is the bridge to the probability simplex.
- Hidden activations *are* model structure, so they are `Layer`s.
- There is no hidden `Sigmoid` layer. Hidden activations are `Tanh` and `ReLU` only. Sigmoid appears solely inside `SigmoidBCE`. No duplication.

This principle ties directly to the conceptual point the library teaches (Section 9): pre-softmax logits are log-probabilities only up to a shared additive constant.

## 4. Architecture

### 4.1 Module layout

A self-contained `nn/` directory, plus two repository-root support files:

| File | Contents |
|---|---|
| `nn/neural_net.py` | Vector helpers; `Layer` base + `Linear`, `Tanh`, `ReLU`; `Loss` base + `SigmoidBCE`, `SoftmaxCrossEntropy`; `Network`; `gradient_check`. Standard library only (`math`, `random`). |
| `nn/demos.py` | Seeded toy-dataset generators; an ASCII decision-boundary plotter; three demos. |
| `nn/test_gradients.py` | Asserts `gradient_check` error is tiny across configurations. Plain `assert`, runnable as `python test_gradients.py`. |
| `nn/visualize.py` | Optional live training visualization (Section 12). The only file permitted third-party libraries. |
| `nn/walkthrough.md` | The pedagogical narrative and derivations. |
| `requirements.txt` | The visualization module's dependencies (`matplotlib`, `pillow`). |
| `.gitignore` | Ignores `__pycache__/` and the generated `training.gif`. |

### 4.2 Vector helpers

Vectors are `list[float]`. The only shared helper that earns a name is:

- `dot(u, v) -> float`

Everything else (outer products, transposed mat-vec) is a one-line comprehension written inline where it is used. The library deliberately does **not** build a matrix type — that would be a miniature matrix library, against the spirit of the exercise.

### 4.3 The `Layer` interface

Every layer implements exactly three methods:

```
forward(x)      -> output vector        # also caches whatever backward needs
backward(g)     -> gradient w.r.t. input  # g = dL/d(output); also stashes param grads
parameters()    -> list of (values, grads) pairs
```

`parameters()` yields pairs of **parallel, flat `list[float]`s**: a mutable parameter vector and its (same-length) gradient vector. The base `Layer` provides a default `parameters()` returning `[]`, so activation layers implement only `forward` and `backward`.

`backward` **accumulates** parameter gradients (`+=`) so a mini-batch can sum per-example gradients. `Network.zero_grad` clears them between batches.

### 4.4 `Linear` — a stack of logistic units

A `Linear` layer with `n_in` inputs and `n_out` outputs stores:

- `weights`: a list of `n_out` weight vectors, each a `list[float]` of length `n_in`. Weight vector `i` is the `i`-th neuron.
- `bias`: a `list[float]` of length `n_out`.
- `dweights`, `dbias`: gradient accumulators of the same shapes.

Forward:

```
y[i] = dot(weights[i], x) + bias[i]      # one logistic-style unit per i
```

Backward, given `g = dL/dy`:

```
dweights[i][j] += g[i] * x[j]            # outer product, accumulated
dbias[i]       += g[i]
dx[j]           = sum_i weights[i][j] * g[i]   # returned to the previous layer
```

`parameters()` yields `(weights[i], dweights[i])` for each neuron `i`, plus `(bias, dbias)`. Every parameter is therefore a flat `list[float]`.

Initialization: `weights` drawn from `uniform(-r, r)` with `r = 1 / sqrt(n_in)` (a Xavier-style scale that keeps activations well-conditioned); `bias` initialized to zero. Uses the `random` module; demos call `random.seed(...)` for reproducibility.

### 4.5 Activations: `Tanh`, `ReLU`

Hidden activations. Each caches what it needs in `forward` and has no parameters.

- `Tanh`: `forward` computes and caches `t = tanh(x)` componentwise; `backward(g)` returns `g[i] * (1 - t[i]**2)`.
- `ReLU`: `forward` caches the input sign; `backward(g)` returns `g[i]` where input `> 0`, else `0`.

### 4.6 The `Loss` interface

Every loss implements three methods:

```
value(logits, y) -> float                # the scalar loss
grad(logits, y)  -> list[float]          # dL/d(logits) — the backward seed
probs(logits)    -> list[float]          # apply the output activation
```

Targets `y` are integer class labels: `y in {0, 1}` for `SigmoidBCE`, `y in {0, ..., K-1}` for `SoftmaxCrossEntropy`.

- **`SigmoidBCE`** — for a network with a single output (`n_out = 1`). `probs` returns `[sigmoid(z)]`; `grad` returns `[p - y]`; `value` is binary cross-entropy.
- **`SoftmaxCrossEntropy`** — for `K >= 2` outputs. `probs` returns `softmax(logits)`; `grad` returns `p - onehot(y)`; `value` is `-log p[y]`.

Both losses have `grad = p - y` (with `y` read as a one-hot target). The code presents them as parallel twins; the walkthrough explains *why* they coincide — binary logistic regression is two-class softmax with one logit pinned to zero.

### 4.7 `Network` — the orchestrator

Constructed from a list of layers and a loss:

```
Network(layers, loss)
```

Methods:

| Method | Behavior |
|---|---|
| `forward(x)` | Run layers in order; cache and return the logits. |
| `backward(y)` | Seed `g = loss.grad(logits, y)`; run layers in reverse, threading `g = layer.backward(g)`. |
| `parameters()` | Concatenation of every layer's `parameters()`. |
| `zero_grad()` | Set every gradient entry to `0` (generic, via `parameters()`). |
| `step(lr, n)` | For each `(v, g)` pair: `v[k] -= (lr / n) * g[k]` (generic, via `parameters()`). `n` is the batch size, so the update uses the *mean* batch gradient. |
| `loss_value(x, y)` | `loss.value(forward(x), y)` — used by demos and the gradient checker. |
| `predict(x)` | `loss.probs(forward(x))` — returns a probability vector. |
| `fit(X, Y, epochs, lr, batch_size=1, verbose=False, callback=None)` | Mini-batch SGD training loop; returns the per-epoch mean-loss history. When `callback` is supplied, calls `callback(epoch, net, history)` at the end of every epoch. |

`step` and `zero_grad` are written **once**, generically, on top of `parameters()`. There is no per-layer `step` and no `Optimizer` class.

## 5. Data Flow

### 5.1 Forward

`Network.forward(x)` passes `x` through each layer in order. Each layer caches whatever its `backward` will need (a `Linear` caches its input; `Tanh` caches its output; `ReLU` caches its input sign). The final layer's output is the logit vector.

### 5.2 Backward (a reverse fold)

```
g = loss.grad(logits, y)            # the famous p - y
for layer in reversed(layers):
    g = layer.backward(g)           # each returns dL/d(its input)
```

Each `layer.backward(g)` does two things: accumulate this layer's parameter gradients, and return the gradient with respect to its input for the previous layer. The chain rule never appears as one big expression — it is the composition of small, local steps.

### 5.3 Training loop (mini-batch SGD)

```
for epoch in range(epochs):
    shuffle the (x, y) pairs
    for each batch of size up to batch_size:
        net.zero_grad()
        for (x, y) in batch:
            net.forward(x)
            net.backward(y)         # accumulates per-example gradients
        net.step(lr, len(batch))    # applies the mean gradient
```

`batch_size` subsumes the special cases: `1` is pure stochastic gradient descent, `len(X)` is full-batch gradient descent. The batch loss is defined as the *mean* per-example loss, so the batch gradient is the mean of per-example gradients — hence `step` divides by `n`.

## 6. Numerical Stability

Handled explicitly; each trick is also a teaching moment.

- **Softmax** subtracts the max logit before exponentiating: `softmax(z) = softmax(z - max z)`. This is *exactly* the additive-constant freedom discussed in Section 9 — softmax is invariant to a shared shift of all logits, and here that freedom is spent to avoid overflow. The walkthrough makes this callback explicit.
- **Cross-entropy** is computed as `logsumexp(z) - z[y]` rather than `-log(softmax(z)[y])`, avoiding `log(0)`.
- **Sigmoid** uses the sign-branched form: `1 / (1 + exp(-z))` for `z >= 0`, and `exp(z) / (1 + exp(z))` for `z < 0`.
- **Binary cross-entropy** is computed from the logit `z` directly via `max(z, 0) - z * y + log(1 + exp(-|z|))`, avoiding `log(0)`.

## 7. Public API — the progression as configuration

```python
# Stage 1 — logistic regression: one Linear layer, sigmoid + BCE
net = Network([Linear(2, 1)], loss=SigmoidBCE())

# Stage 2 — multi-class (softmax) regression: one Linear layer, softmax + cross-entropy
net = Network([Linear(2, 3)], loss=SoftmaxCrossEntropy())

# Stage 3 — MLP: stacked layers, identical loss, identical training
net = Network([Linear(2, 8), Tanh(),
               Linear(8, 3)], loss=SoftmaxCrossEntropy())

net.fit(X, Y, epochs=200, lr=0.1, batch_size=16)
net.predict(x)        # -> probability vector
```

## 8. Gradient Checking

`gradient_check(net, x, y, eps=1e-5) -> float` verifies the analytical gradients against central finite differences and returns the worst relative error found.

Procedure:

1. `net.zero_grad()`, then `net.forward(x)` and `net.backward(y)` to populate the analytical gradients.
2. For every `(values, grads)` pair from `net.parameters()`, and every index `k`:
   - Perturb `values[k]` by `+eps` and `-eps`, evaluating `net.loss_value(x, y)` each time.
   - Numerical gradient: `(L_plus - L_minus) / (2 * eps)`.
   - Relative error: `|numerical - grads[k]| / max(|numerical| + |grads[k]|, tiny)`.
   - Restore `values[k]`.
3. Return the maximum relative error.

`test_gradients.py` asserts this error is below a small tolerance for: logistic regression, softmax regression, a `Tanh` MLP, and a `ReLU` MLP.

**ReLU-kink caveat:** ReLU is not differentiable at `0`. A finite-difference straddling the kink disagrees with the analytical sub-gradient. With random initialization and random data the probability of a pre-activation landing exactly on `0` is effectively zero, so the check passes in practice; the strict tolerance is anchored on the `Tanh` network, and the caveat is explained in the walkthrough.

## 9. Math the Walkthrough Must Capture

These derivations are recorded here so the implementation is unambiguous; they are explained in full in `walkthrough.md`.

- **Logits as unnormalized log-probabilities.** With `p = softmax(z)`, `log p_i = z_i - logsumexp(z)`. The logits equal the log-probabilities plus a shared additive constant `logsumexp(z)` — constant across classes for a given input, but varying across inputs. Softmax is invariant under `z -> z + c`, so training cannot pin down that constant; it fixes only the differences `z_i - z_j = log(p_i / p_j)`, the log-odds. A non-negative activation on the logits (e.g. ReLU) would be impossible to read as raw log-probabilities — log-probabilities are non-positive — which is exactly why the additive constant is necessary.
- **BCE gradient.** With `p = sigmoid(z)` and binary cross-entropy loss, `dL/dz = p - y`.
- **Softmax cross-entropy gradient.** With `p = softmax(z)` and loss `-log p[y]`, `dL/dz_i = p_i - [i == y]`.
- **`Linear` gradients.** For `y = Wx + b` and incoming `g = dL/dy`: `dL/dW[i][j] = g[i] * x[j]`, `dL/db[i] = g[i]`, `dL/dx[j] = sum_i W[i][j] * g[i]`.
- **Activation gradients.** `Tanh`: `dL/dx = g * (1 - tanh(x)**2)`. `ReLU`: `dL/dx = g` where `x > 0`, else `0`.
- **Per-layer chain rule.** Backprop is the composition of these local gradients, applied to layers in reverse. Depth costs no new math — only more composition.

## 10. Demos

`nn/demos.py` contains seeded dataset generators, an ASCII plotter, and three demos.

- **Dataset generators** (using a seeded `random`): two 2-D Gaussian blobs (binary), three 2-D Gaussian blobs (3-class), and XOR (the four canonical points, optionally a noisy cloud around them).
- **ASCII decision-boundary plotter.** For a 2-D problem, sample a grid of points, call `net.predict`, take the argmax, and print one character per predicted class. A dependency-free way to *see* what the network learned.
- **Demo 1 — logistic regression** on two blobs: train, print accuracy, render the linear decision boundary.
- **Demo 2 — softmax regression** on three blobs: train, print accuracy, render the three-way boundary.
- **Demo 3 — MLP on XOR:** first show that a one-`Linear`-layer network (logistic regression) *cannot* separate XOR, then show that adding a `Tanh` hidden layer solves it. The decisive demonstration of why hidden layers exist.

## 11. Walkthrough Outline

`nn/walkthrough.md` follows the progression:

1. **Logistic regression** — the sigmoid, binary cross-entropy, the `p - y` gradient, gradient descent.
2. **Multi-class via softmax** — softmax, cross-entropy, the same `p - y` gradient; binary logistic as the two-class special case.
3. **Logits vs. log-probabilities** — the additive-constant insight; why logits are unnormalized log-probabilities; the ReLU-on-logits argument.
4. **Stacking into an MLP** — hidden layers, hidden activations, per-layer forward/backward; the chain rule as local composition; "no new math, only composition."
5. **Numerical stability as gauge freedom** — the softmax max-subtraction *is* the additive-constant freedom from section 3, reused.
6. **Verifying gradients** — numerical gradients, the central-difference checker, the ReLU-kink caveat.
7. **Closing note: from per-layer to per-operation** — if every scalar operation (not just every layer) carried its own local backward, per-layer backprop would become full automatic differentiation. This is what autograd engines do; it is the natural generalization, and it is left as a pointer rather than built.

## 12. Visualization Component

An optional module, `nn/visualize.py`, for *watching* a network train. It is
the one component permitted third-party libraries — matplotlib, and Pillow
for the GIF fallback. The core library is untouched by it.

### 12.1 The seam: a per-epoch callback

`Network.fit` takes an optional `callback` argument. When supplied, `fit`
calls `callback(epoch, net, history)` at the end of every epoch: `epoch` is
the 0-based index, `net` is the network, `history` is the loss list so far. A
callback is an ordinary function, so this adds no dependency to the core — it
is the only change `neural_net.py` needs. Everything visual lives in
`visualize.py`.

### 12.2 `watch_training`

`watch_training(net, X, Y, epochs, lr, frames=50, gif_path="training.gif",
**fit_kwargs)` trains `net` on 2-D data while rendering a two-panel
matplotlib figure: the left panel shows the decision boundary (predicted
class over a grid, with the data points overlaid), the right panel shows the
loss curve. It installs an internal callback that redraws both panels,
throttled to roughly `frames` updates across the run
(`every = max(1, epochs // frames)`). Extra keyword arguments pass through to
`fit`. It assumes 2-D inputs — every demo dataset is 2-D.

### 12.3 Live or saved — chosen automatically

`watch_training` inspects matplotlib's active backend:

- **Interactive backend available** — it updates a live window in place as
  training runs, and shows the final figure.
- **No display (headless)** — it captures a frame at each throttled update
  and writes an animated GIF to `gif_path` after training, using Pillow.

Running `python nn/visualize.py` trains a 2-8-1 `Tanh` MLP on XOR through
`watch_training` — the live counterpart of the ASCII XOR demo.

## 13. Deliverables Summary

- `nn/neural_net.py` — the unified library.
- `nn/demos.py` — datasets, ASCII plotter, three demos.
- `nn/test_gradients.py` — gradient-check assertions.
- `nn/visualize.py` — optional matplotlib training visualization.
- `nn/walkthrough.md` — the pedagogical narrative.
- `requirements.txt` and `.gitignore` — visualization dependencies and ignore rules.
