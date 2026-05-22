# A Neural Network from Scratch — Walkthrough

This library builds a neural network in pure Python — no NumPy, no matrix
libraries. It exists to make one progression concrete: logistic regression,
then multi-class (softmax) regression, then the multi-layer perceptron. All
three are configurations of a single `Network` class.

## 1. Logistic regression

A logistic-regression model scores an input `x` with a weighted sum, the
**logit**:

    z = w . x + b

It squashes that score into a probability with the **sigmoid**:

    p = sigmoid(z) = 1 / (1 + e^(-z))

`p` is the model's estimate of `P(y = 1 | x)`. To train it we need a loss
that is small when `p` is close to the label `y` in {0, 1}. That is **binary
cross-entropy**:

    L = -[ y log p + (1 - y) log(1 - p) ]

The gradient of this loss with respect to the logit `z` is remarkably clean:

    dL/dz = p - y

The prediction minus the target. Every weight gradient follows by the chain
rule, `dL/dw_j = (p - y) x_j`, and we descend: `w <- w - lr * dL/dw`.

In code, logistic regression is one `Linear` layer with a `SigmoidBCE` loss:

    net = Network([Linear(n_features, 1)], loss=SigmoidBCE())

## 2. Multi-class regression via softmax

With `K` classes the model produces `K` logits — one weighted sum per class.
We turn the logit vector `z` into a probability distribution with the
**softmax**:

    p_i = e^(z_i) / sum_j e^(z_j)

The matching loss is **categorical cross-entropy**, `L = -log p_y`, where `y`
is the index of the correct class. Its gradient with respect to the logits
is — again —

    dL/dz_i = p_i - [i = y]

that is, `p - y` with `y` read as a one-hot vector. The same expression as
the binary case. That is no coincidence: binary logistic regression *is*
two-class softmax with one logit pinned to zero (softmax depends only on
logit differences, so one logit may be fixed freely). The binary sigmoid is
the K=2 softmax in disguise.

In code, softmax regression is — again — one `Linear` layer:

    net = Network([Linear(n_features, K)], loss=SoftmaxCrossEntropy())

## 3. Logits are log-probabilities — up to a constant

Take the log of the softmax:

    log p_i = z_i - log sum_j e^(z_j)

So the logits are *not* the log-probabilities. They are the log-probabilities
plus a shared term `log sum_j e^(z_j)` — the same for every class `i`. The
logits are **unnormalized log-probabilities**.

That constant cannot be recovered from `p`. Softmax is unchanged if you add
any constant `c` to every logit: `e^(z_i + c)` has a factor `e^c` on top and
bottom, and it cancels. Training therefore can never pin the constant down.
What training *does* pin down is logit **differences**:

    z_i - z_j = log p_i - log p_j = log(p_i / p_j)

the log-odds between classes. The constant is gauge — free, unobservable.

One sharp way to see that logits are not raw log-probabilities: a probability
is at most 1, so a log-probability is at most 0. But nothing stops a
network's logits from being positive — indeed, if you put a ReLU just before
the output, every logit is at least 0. They still describe a perfectly good
distribution, because the additive constant absorbs the offset. Logits live
on a different scale than log-probabilities; only their differences are
shared between them.

## 4. Stacking into a multi-layer perceptron

A `Linear` layer is a list of units, each a logistic-regression-style weight
vector. The multi-layer perceptron simply stacks `Linear` layers with a
non-linear **activation** between them (`Tanh` or `ReLU`). Without the
non-linearity, stacked linear maps collapse into a single linear map and buy
nothing.

    net = Network([Linear(2, 8), Tanh(), Linear(8, 3)],
                  loss=SoftmaxCrossEntropy())

Training needs the gradient of the loss with respect to every weight in every
layer. The trick — **backpropagation** — is to make the chain rule *local*.
Each layer implements two methods for the chain rule:

    forward(x)  -> output
    backward(g) -> gradient w.r.t. its input   (g = gradient w.r.t. its output)

`backward` also stashes the layer's own parameter gradients — a third method,
`parameters()`, later hands those weights and their gradients to the optimizer
and the gradient checker. A layer never sees the rest of the network. The
`Network` runs `forward` through the layers in order, then `backward` through
them in reverse, threading the gradient:

    g = loss.grad(logits, y)          # the familiar p - y
    for layer in reversed(layers):
        g = layer.backward(g)

Each layer's `backward` is tiny. For `Linear` (`y = W x + b`): the parameter
gradients are `dL/dW_ij = g_i x_j` and `dL/db_i = g_i`, and the gradient
passed back is `dL/dx_j = sum_i W_ij g_i`. For `Tanh`: `g * (1 - tanh(x)^2)`.
For `ReLU`: `g` where the input was positive, else 0.

The key realization: the MLP introduced **no new math**. The `p - y` you
derived for logistic regression is exactly the output layer's `backward`.
Depth is just more composition of the same local steps.

## 5. Numerical stability is the gauge freedom, reused

`e^z` overflows for moderately large `z`. The fix for softmax: subtract the
largest logit first,

    softmax(z) = softmax(z - max z)

This is *exactly* the additive-constant freedom from section 3. Softmax does
not care about a shared shift of the logits, so we spend that freedom to keep
the exponentials in a safe range. Stability here is not a new idea bolted on
— it is the gauge symmetry put to work.

Cross-entropy is computed as `logsumexp(z) - z_y` rather than
`-log(softmax(z)_y)`, which would risk `log(0)`. The sigmoid is evaluated
with a sign branch, and binary cross-entropy is computed straight from the
logit, both to avoid overflow and `log(0)`.

## 6. Trust, but verify: numerical gradients

Hand-derived gradients are easy to get subtly wrong. There is a slow but
foolproof check: the definition of a derivative itself. For any parameter
`t`,

    dL/dt ~= (L(t + eps) - L(t - eps)) / (2 eps)

`gradient_check` computes this central difference for every parameter and
compares it to the analytical gradient from `backward`. If they disagree, the
analytical gradient is wrong. This is far too slow to train with — one
forward pass per parameter — but it is the ground truth that keeps the fast
method honest. `test_gradients.py` runs it on all four configurations.

One caveat: `ReLU` has a kink at 0 and no derivative there. A finite
difference that straddles the kink will disagree with the analytical
sub-gradient. With random weights the probability of a pre-activation landing
exactly on the kink is effectively zero, so the check passes in practice —
but it is a real reminder that `ReLU` is not differentiable everywhere.

## 7. Closing note: from per-layer to per-operation

Backpropagation here is per-*layer*: each layer carries a local `backward`.
But nothing forced the unit to be a layer. If every scalar *operation* —
every `+`, every `*`, every `tanh` — carried its own local backward, the same
reverse pass would compute gradients through an arbitrary expression, with no
hand-derived layer gradients at all. That is **automatic differentiation**,
and it is what every modern framework's autograd engine does. It is the same
idea as this library, at a finer grain. Building one is the natural next
step.
