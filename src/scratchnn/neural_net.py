"""A pure-Python pedagogical neural network library.

Standard library only. Teaches logistic regression, softmax regression, and
the multi-layer perceptron as configurations of one Network class. The
network computes logits; the loss interprets them.
"""
import math
import random


# --------------------------------------------------------------------------
# Vector helpers
# --------------------------------------------------------------------------

def dot(u, v):
    """Dot product of two equal-length vectors."""
    return sum(ui * vi for ui, vi in zip(u, v))


# --------------------------------------------------------------------------
# Numeric primitives
# --------------------------------------------------------------------------

def sigmoid(z):
    """Logistic sigmoid, evaluated without overflow for large |z|."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logsumexp(zs):
    """log(sum(exp(zs))), shifted by max(zs) to avoid overflow."""
    m = max(zs)
    return m + math.log(sum(math.exp(z - m) for z in zs))


def softmax(zs):
    """Softmax of a vector, max-subtracted for numerical stability."""
    m = max(zs)
    exps = [math.exp(z - m) for z in zs]
    total = sum(exps)
    return [e / total for e in exps]


def softplus(z):
    """Softplus: log(1 + exp(z)). The smooth analogue of the positive part.

    Evaluated as max(z, 0) + log1p(exp(-|z|)) to avoid overflow for large |z|.
    Its derivative is the logistic sigmoid, sigmoid(z).
    """
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

class Layer:
    """Base class for network layers.

    A layer maps an input vector to an output vector (forward) and
    propagates a gradient back to its input (backward). Layers with
    trainable parameters expose them through parameters().
    """

    def forward(self, x):
        raise NotImplementedError

    def backward(self, g):
        """Given g = dL/d(output), return dL/d(input).

        Layers with parameters also accumulate their parameter gradients.
        """
        raise NotImplementedError

    def parameters(self):
        """Return a list of (values, grads) pairs.

        Each pair is a mutable parameter vector and its same-length
        gradient vector. The base layer has no parameters.
        """
        return []


class Linear(Layer):
    """An affine map y = W x + b, stored as one weight vector per unit."""

    def __init__(self, n_in, n_out):
        r = 1.0 / math.sqrt(n_in)
        self.weights = [[random.uniform(-r, r) for _ in range(n_in)]
                        for _ in range(n_out)]
        self.bias = [0.0 for _ in range(n_out)]
        self.dweights = [[0.0 for _ in range(n_in)] for _ in range(n_out)]
        self.dbias = [0.0 for _ in range(n_out)]
        self.x = None

    def forward(self, x):
        self.x = x
        return [dot(w, x) + b for w, b in zip(self.weights, self.bias)]

    def backward(self, g):
        for i, gi in enumerate(g):
            for j, xj in enumerate(self.x):
                self.dweights[i][j] += gi * xj
            self.dbias[i] += gi
        return [sum(self.weights[i][j] * g[i] for i in range(len(g)))
                for j in range(len(self.x))]

    def parameters(self):
        pairs = list(zip(self.weights, self.dweights))
        pairs.append((self.bias, self.dbias))
        return pairs


class Tanh(Layer):
    """Hyperbolic-tangent activation, applied componentwise."""

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = [math.tanh(xi) for xi in x]
        return self.out

    def backward(self, g):
        return [gi * (1.0 - oi * oi) for gi, oi in zip(g, self.out)]


class ReLU(Layer):
    """Rectified-linear activation, applied componentwise."""

    def __init__(self):
        self.positive = None

    def forward(self, x):
        self.positive = [xi > 0.0 for xi in x]
        return [xi if xi > 0.0 else 0.0 for xi in x]

    def backward(self, g):
        return [gi if p else 0.0 for gi, p in zip(g, self.positive)]


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

class Loss:
    """Base class for losses.

    A loss interprets the network's logits: it produces a scalar loss
    (value), the gradient of that loss w.r.t. the logits (grad), and the
    probability vector the logits encode (probs).
    """

    def value(self, logits, y):
        raise NotImplementedError

    def grad(self, logits, y):
        raise NotImplementedError

    def probs(self, logits):
        raise NotImplementedError


class SigmoidBCE(Loss):
    """Sigmoid output with binary cross-entropy. Expects a single logit."""

    def value(self, logits, y):
        z = logits[0]
        # Stable BCE straight from the logit:
        #   max(z, 0) - z * y + log(1 + exp(-|z|))
        return max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))

    def grad(self, logits, y):
        return [sigmoid(logits[0]) - y]

    def probs(self, logits):
        return [sigmoid(logits[0])]


class SoftmaxCrossEntropy(Loss):
    """Softmax output with categorical cross-entropy. Expects K >= 2 logits."""

    def value(self, logits, y):
        # Stable cross-entropy: -log(softmax(z)[y]) = logsumexp(z) - z[y]
        return logsumexp(logits) - logits[y]

    def grad(self, logits, y):
        p = softmax(logits)
        p[y] -= 1.0
        return p

    def probs(self, logits):
        return softmax(logits)


class MSELoss(Loss):
    """Identity output with mean squared error.

    The raw network output is the prediction (identity link function).
    Targets are list[float] of the same length as the output. The 1/2
    convention makes the gradient match the canonical p - y pattern of
    the other losses: dL/dz_i = z_i - y_i.
    """

    def value(self, logits, y):
        return 0.5 * sum((zi - yi) ** 2 for zi, yi in zip(logits, y))

    def grad(self, logits, y):
        return [zi - yi for zi, yi in zip(logits, y)]

    def probs(self, logits):
        # Identity link: the prediction IS the raw output.
        return list(logits)


class PoissonNLLLoss(Loss):
    """Log-link Poisson negative log-likelihood. Expects a single logit.

    The raw output z is interpreted as a log-rate. The predicted rate is
    lambda = exp(z), guaranteed positive. Target y is a count (typically a
    non-negative integer, but any real is accepted). The constant log(y!)
    is omitted; it does not depend on z and does not affect optimization.

    Canonical-link form: dL/dz = exp(z) - y = lambda - y.
    """

    def value(self, logits, y):
        z = logits[0]
        return math.exp(z) - z * y

    def grad(self, logits, y):
        z = logits[0]
        return [math.exp(z) - y]

    def probs(self, logits):
        # Inverse of log link: the predicted rate is exp(z).
        return [math.exp(logits[0])]


class GaussianNLLLoss(Loss):
    """Heteroscedastic Gaussian NLL. Expects two logits, (z_mu, z_s).

    The first logit is the mean mu = z_mu (identity link on the mean). The
    second logit z_s is mapped through softplus to a positive scale,
    sigma = softplus(z_s), enforcing positivity without clamping. Target y
    is a single real number. The constant 0.5 * log(2*pi) is omitted.

    Loss: 0.5 * (y - mu)**2 / sigma**2 + log(sigma).
    Gradients are hand-derived; the chain through softplus uses
    d(softplus(z_s))/dz_s = sigmoid(z_s).
    """

    def value(self, logits, y):
        mu = logits[0]
        s = softplus(logits[1])
        return 0.5 * (y - mu) ** 2 / (s * s) + math.log(s)

    def grad(self, logits, y):
        mu = logits[0]
        z_s = logits[1]
        s = softplus(z_s)
        dmu = (mu - y) / (s * s)
        ds = -((y - mu) ** 2) / (s ** 3) + 1.0 / s
        return [dmu, ds * sigmoid(z_s)]

    def probs(self, logits):
        # Report (mu, sigma) for the predictive Gaussian.
        return [logits[0], softplus(logits[1])]


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class Network:
    """A stack of layers plus a loss.

    The layers map an input to logits; the loss interprets those logits.
    """

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.logits = None

    def forward(self, x):
        """Run the layers in order; cache and return the logits."""
        for layer in self.layers:
            x = layer.forward(x)
        self.logits = x
        return x

    def backward(self, y):
        """Backpropagate from the loss, accumulating parameter gradients."""
        g = self.loss.grad(self.logits, y)
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def parameters(self):
        """Every (values, grads) pair across all layers."""
        pairs = []
        for layer in self.layers:
            pairs.extend(layer.parameters())
        return pairs

    def zero_grad(self):
        """Reset every parameter gradient to zero."""
        for _values, grads in self.parameters():
            for k in range(len(grads)):
                grads[k] = 0.0

    def step(self, lr, n):
        """Apply one SGD update using the mean gradient over n examples."""
        for values, grads in self.parameters():
            for k in range(len(values)):
                values[k] -= (lr / n) * grads[k]

    def loss_value(self, x, y):
        """Scalar loss for a single (x, y) example."""
        return self.loss.value(self.forward(x), y)

    def predict(self, x):
        """Probability vector for input x."""
        return self.loss.probs(self.forward(x))

    def fit(self, X, Y, epochs, lr, batch_size=1, verbose=False, callback=None):
        """Train with mini-batch SGD; return the per-epoch mean-loss history.

        If callback is given, it is called as callback(epoch, net, history)
        at the end of every epoch.
        """
        history = []
        data = list(zip(X, Y))
        for epoch in range(epochs):
            random.shuffle(data)
            total = 0.0
            for start in range(0, len(data), batch_size):
                batch = data[start:start + batch_size]
                self.zero_grad()
                for x, y in batch:
                    self.forward(x)
                    total += self.loss.value(self.logits, y)
                    self.backward(y)
                self.step(lr, len(batch))
            mean_loss = total / len(data)
            history.append(mean_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0
                            or epoch == epochs - 1):
                print(f"  epoch {epoch:4d}   loss {mean_loss:.4f}")
            if callback is not None:
                callback(epoch, self, history)
        return history


# --------------------------------------------------------------------------
# Gradient checking
# --------------------------------------------------------------------------

def gradient_check(net, x, y, eps=1e-5):
    """Worst relative error between analytical and numerical gradients.

    The analytical gradients come from one forward/backward pass; each is
    compared against a central finite difference of the loss.
    """
    net.zero_grad()
    net.forward(x)
    net.backward(y)
    worst = 0.0
    for values, grads in net.parameters():
        for k in range(len(values)):
            original = values[k]
            values[k] = original + eps
            loss_plus = net.loss_value(x, y)
            values[k] = original - eps
            loss_minus = net.loss_value(x, y)
            values[k] = original
            numerical = (loss_plus - loss_minus) / (2.0 * eps)
            analytical = grads[k]
            denom = max(abs(numerical) + abs(analytical), 1e-12)
            worst = max(worst, abs(numerical - analytical) / denom)
    return worst
