# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`scratchnn` is a **pedagogical** neural network library in pure Python (`math`, `random` only), written to be *read*. The entire core lives in one ~300-line module: `src/scratchnn/neural_net.py`. Treat clarity as a hard design constraint. Cleverness, premature abstraction, and "make it more like PyTorch" changes are off-spec.

Companion docs:
- `README.md`: install, quick-start, layout.
- `docs/design.md`: full design rationale, math, and non-goals. Long, and written before packaging, so it describes the old `nn/` layout instead of `src/scratchnn/`. The rationale is unchanged.
- `docs/walkthrough.md`: pedagogical narrative.

## Commands

```bash
pip install -e .                 # core (standard library only)
pip install -e ".[viz]"          # adds matplotlib + Pillow for the optional viz

python tests/test_neural_net.py  # 33 unit tests
python tests/test_gradients.py   # 4 analytic-vs-finite-difference checks (tol 1e-4)
python examples/demos.py         # logistic / softmax / XOR demos with ASCII boundaries
python -m scratchnn.visualize    # live (or headless-GIF) XOR MLP training
```

Tests are plain `assert` scripts (no pytest). Each test file ends with a runner that auto-discovers `test_*` functions in its module globals. To run a single test after `pip install -e .`:

```bash
python -c "import sys; sys.path.insert(0,'tests'); from test_neural_net import test_dot; test_dot()"
```

## Architecture invariants

These are load-bearing for the library's pedagogy. Preserve them.

### Network produces logits; Loss interprets them

A `Network` produces raw, unnormalized scores. The output activation (sigmoid, softmax) belongs to the `Loss`, not the model. Consequences:

- There is **no `Sigmoid` layer**. Hidden activations are `Tanh` and `ReLU` only; sigmoid lives inside `SigmoidBCE`.
- `Network.predict` is `loss.probs(forward(x))`. Never compute probabilities inside `Network`.
- A new output activation means a new `Loss`, not a new `Layer`.

### Layer protocol: three methods, flat `(values, grads)` pairs

```
forward(x)   -> output vector            # also caches whatever backward needs
backward(g)  -> dL/d(input)              # also accumulates parameter gradients (+=)
parameters() -> list of (values, grads)  # parallel flat list[float] pairs
```

Parameters are always pairs of **parallel flat `list[float]`s** (e.g. `Linear` yields one pair per neuron's weight vector, plus one pair for the bias). `zero_grad` and `step` are written **once, generically**, on top of `parameters()` (see `neural_net.py:225-235`). There is no per-layer `step` and no `Optimizer` class. Don't add one.

### Backward accumulates; `zero_grad` clears; `step` divides by `n`

`backward` uses `+=` so a mini-batch can sum per-example gradients. `step(lr, n)` then divides by `n` to apply the *mean* batch gradient. Batch loss is defined as mean per-example, so the gradient is too. `batch_size=1` is pure SGD, `batch_size=len(X)` is full-batch.

### Vectors are `list[float]`. There is no matrix type.

`dot()` is the only named vector helper. Outer products, transposed mat-vec, etc., are one-line comprehensions inline. Building a matrix abstraction would be a miniature numerics library, against the spirit of the project.

### Numerical stability is hand-derived and explicit

Don't "simplify" these without thinking. They are deliberate, and each is a teaching moment in the walkthrough:

- Softmax / cross-entropy: max-subtraction, computed as `logsumexp(z) - z[y]`, not `-log softmax(z)[y]`.
- Sigmoid: sign-branched (`1/(1+e^-z)` vs `e^z/(1+e^z)`).
- BCE: stable from-logit form `max(z,0) - z*y + log1p(exp(-|z|))`.

## When changing the math

After any change to a `forward`, `backward`, or `Loss.grad`, run `tests/test_gradients.py`. If you add a new `Layer` or `Loss`, add a `gradient_check` case for it.

ReLU is not differentiable at 0; the strict `1e-4` tolerance still holds because random init and inputs don't land on the kink. Don't relax the tolerance to mask a real bug. Anchor a new check on `Tanh` instead, or fix the seed.

## Deliberately out of scope

No NumPy, no autograd engine, no `Optimizer` class, no momentum or Adam, no regularization, no LR schedules, no save/load, no GPU, no batching speedups. Each of these exclusions keeps the reader's attention on the network and on backpropagation. If a change wants to add any of them, it likely doesn't belong here. The walkthrough's closing note is the precedent for *mentioning* the natural generalization (autograd) without building it.

## The viz module is a quarantined exception

`src/scratchnn/visualize.py` is the only file allowed third-party dependencies (matplotlib, Pillow). It hooks into the core *only* through `Network.fit(..., callback=...)`. The per-epoch callback is the entire seam. Don't pull viz concerns into `neural_net.py`.
