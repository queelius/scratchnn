"""Digit classification on the UCI 8x8 optdigits dataset.

The dataset is ~3800 training plus ~1800 test samples of 8x8 grayscale digits,
with integer pixel values 0-16. Each value is the count of "on" pixels in a
4x4 block of the original 32x32 bitmap.

We train two configurations of scratchnn on the same data:

  1. Softmax regression: one Linear layer, no hidden units.
  2. MLP:                one Linear, Tanh, one Linear.

The walkthrough threads the resulting accuracies through its progression, and
the gap between the two motivates the inductive-bias discussion.

Run: python examples/digits.py
"""
import os
import random

import scratchnn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "data", "optdigits.tra")
TEST = os.path.join(HERE, "data", "optdigits.tes")


def load_digits(path):
    """Load 8x8 digits from a UCI optdigits CSV file.

    Each line is 64 integer features (0-16) followed by a class label (0-9).
    Returns (X, Y): X is list[list[float]] with values in [0, 1]; Y is list[int].
    """
    X, Y = [], []
    with open(path) as f:
        for line in f:
            parts = [int(p) for p in line.strip().split(",")]
            X.append([p / 16.0 for p in parts[:64]])
            Y.append(parts[64])
    return X, Y


def predicted_label(probs):
    return max(range(len(probs)), key=lambda i: probs[i])


def accuracy(model, X, Y):
    hits = sum(predicted_label(model.predict(x)) == y for x, y in zip(X, Y))
    return hits / len(X)


def render_ascii_digit(features, label):
    """Print one 8x8 digit as ASCII alongside its label."""
    chars = " .:-=+*#%@"
    n = len(chars) - 1
    print(f"label = {label}")
    for r in range(8):
        row = features[r * 8:(r + 1) * 8]
        line = "".join(chars[min(int(v * n), n)] for v in row)
        print(f"  {line}")


def train_and_report(model, X_tr, Y_tr, X_te, Y_te, epochs, lr, batch_size,
                     label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    history = model.fit(X_tr, Y_tr, epochs=epochs, lr=lr,
                        batch_size=batch_size, verbose=True)
    tr = accuracy(model, X_tr, Y_tr)
    te = accuracy(model, X_te, Y_te)
    print(f"  train accuracy: {tr:.4f}")
    print(f"  test  accuracy: {te:.4f}")
    return history, tr, te


def main(epochs_softmax=20, epochs_mlp=30, hidden=32, batch_size=32, lr=0.5,
         seed=0):
    print("Loading digits from UCI optdigits...")
    X_tr, Y_tr = load_digits(TRAIN)
    X_te, Y_te = load_digits(TEST)
    print(f"  train: {len(X_tr)} samples")
    print(f"  test:  {len(X_te)} samples")

    print("\nExample digit (training index 0):")
    render_ascii_digit(X_tr[0], Y_tr[0])

    random.seed(seed)
    softmax_model = nn.Network([nn.Linear(64, 10)], nn.SoftmaxCrossEntropy())
    train_and_report(softmax_model, X_tr, Y_tr, X_te, Y_te,
                     epochs=epochs_softmax, lr=lr, batch_size=batch_size,
                     label="Softmax regression (one Linear, no hidden layer)")

    random.seed(seed)
    mlp = nn.Network([nn.Linear(64, hidden), nn.Tanh(),
                      nn.Linear(hidden, 10)], nn.SoftmaxCrossEntropy())
    train_and_report(mlp, X_tr, Y_tr, X_te, Y_te,
                     epochs=epochs_mlp, lr=lr, batch_size=batch_size,
                     label=f"MLP (64-{hidden}-10, Tanh)")


if __name__ == "__main__":
    main()
