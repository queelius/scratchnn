"""Runnable demos for the scratchnn library.

Run: python examples/demos.py
"""
import random

import scratchnn as nn


# --- toy datasets ---------------------------------------------------------

def make_blobs(n_per_class, centers, spread, seed):
    """Gaussian blobs around each center; one integer label per center."""
    rng = random.Random(seed)
    X, Y = [], []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_class):
            X.append([rng.gauss(cx, spread), rng.gauss(cy, spread)])
            Y.append(label)
    return X, Y


def make_xor():
    """The four canonical XOR points."""
    return [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 1, 1, 0]


# --- prediction helpers ---------------------------------------------------

def predicted_label(probs):
    """Class index from a probability vector.

    A SigmoidBCE network outputs a single probability P(class 1); a softmax
    network outputs one probability per class.
    """
    if len(probs) == 1:
        return 1 if probs[0] >= 0.5 else 0
    return max(range(len(probs)), key=lambda i: probs[i])


def accuracy(model, X, Y):
    """Fraction of examples classified correctly."""
    hits = sum(predicted_label(model.predict(x)) == y for x, y in zip(X, Y))
    return hits / len(X)


# --- ASCII visualization --------------------------------------------------

def plot_decision_boundary(model, x_range, y_range, cols=46, rows=22):
    """Print the model's predicted class over a 2-D region as an ASCII grid."""
    symbols = ".ox* "
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    for r in range(rows):
        y = y_hi - (y_hi - y_lo) * (r + 0.5) / rows
        line = []
        for c in range(cols):
            x = x_lo + (x_hi - x_lo) * (c + 0.5) / cols
            label = predicted_label(model.predict([x, y]))
            line.append(symbols[label % len(symbols)])
        print("".join(line))


# --- demos ----------------------------------------------------------------

def demo_logistic_regression():
    print("=" * 60)
    print("Demo 1 -- logistic regression: one Linear layer, sigmoid + BCE")
    print("=" * 60)
    X, Y = make_blobs(60, [(-2.0, -2.0), (2.0, 2.0)], spread=1.1, seed=1)
    random.seed(1)
    model = nn.Network([nn.Linear(2, 1)], nn.SigmoidBCE())
    model.fit(X, Y, epochs=80, lr=0.3, batch_size=12, verbose=True)
    print(f"accuracy: {accuracy(model, X, Y):.3f}")
    plot_decision_boundary(model, (-5.0, 5.0), (-5.0, 5.0))
    print()


def demo_softmax_regression():
    print("=" * 60)
    print("Demo 2 -- softmax regression: one Linear layer, softmax + CE")
    print("=" * 60)
    X, Y = make_blobs(60, [(-3.0, -2.0), (3.0, -2.0), (0.0, 3.0)],
                      spread=1.1, seed=2)
    random.seed(2)
    model = nn.Network([nn.Linear(2, 3)], nn.SoftmaxCrossEntropy())
    model.fit(X, Y, epochs=120, lr=0.3, batch_size=12, verbose=True)
    print(f"accuracy: {accuracy(model, X, Y):.3f}")
    plot_decision_boundary(model, (-7.0, 7.0), (-6.0, 6.0))
    print()


def demo_mlp_xor():
    print("=" * 60)
    print("Demo 3 -- XOR: why hidden layers exist")
    print("=" * 60)
    X, Y = make_xor()

    print("\nOne Linear layer (logistic regression) cannot solve XOR:")
    random.seed(0)
    linear = nn.Network([nn.Linear(2, 1)], nn.SigmoidBCE())
    linear.fit(X, Y, epochs=500, lr=0.5, batch_size=4)
    for x, y in zip(X, Y):
        print(f"  {x} target {y}  ->  p(class 1) = {linear.predict(x)[0]:.3f}")
    print("  every point collapses to p ~ 0.5 -- a line cannot separate XOR")

    print("\nAdd one Tanh hidden layer and XOR becomes solvable:")
    random.seed(0)
    mlp = nn.Network([nn.Linear(2, 8), nn.Tanh(),
                      nn.Linear(8, 1)], nn.SigmoidBCE())
    mlp.fit(X, Y, epochs=4000, lr=0.5, batch_size=4)
    for x, y in zip(X, Y):
        print(f"  {x} target {y}  ->  p(class 1) = {mlp.predict(x)[0]:.3f}")
    print(f"  accuracy: {accuracy(mlp, X, Y):.3f}")
    print()


if __name__ == "__main__":
    demo_logistic_regression()
    demo_softmax_regression()
    demo_mlp_xor()
