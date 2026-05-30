"""Regression demo: fit y = sin(x) with a small MLP and MSE loss.

The architecture is identical to the classification MLPs elsewhere in the
demos. Only the output head and loss differ: identity link, MSE.

Run: python examples/regression.py
"""
import math
import random

import scratchnn as nn


def make_sine_data(n, noise=0.05, seed=0):
    """Sample n points (x, sin(x) + Gaussian noise) on [-pi, pi]."""
    rng = random.Random(seed)
    X = [[rng.uniform(-math.pi, math.pi)] for _ in range(n)]
    Y = [[math.sin(x[0]) + rng.gauss(0.0, noise)] for x in X]
    return X, Y


def render_ascii_curve(model, x_range=(-math.pi, math.pi), cols=60, rows=18):
    """Print the learned curve over a 1-D range as an ASCII plot."""
    x_lo, x_hi = x_range
    y_lo, y_hi = -1.3, 1.3
    grid = [[" "] * cols for _ in range(rows)]

    def to_row(y):
        return min(rows - 1, max(0, int((y_hi - y) / (y_hi - y_lo) * rows)))

    # Plot true sine ('.') and predicted ('*').
    for c in range(cols):
        x = x_lo + (x_hi - x_lo) * c / (cols - 1)
        true_y = math.sin(x)
        pred_y = model.predict([x])[0]
        grid[to_row(true_y)][c] = "."
        grid[to_row(pred_y)][c] = "*"

    print("  . = sin(x),  * = learned MLP")
    for row in grid:
        print("  " + "".join(row))


def main(n=200, hidden=32, epochs=2000, lr=0.05, batch_size=10, seed=0):
    random.seed(seed)
    X, Y = make_sine_data(n, noise=0.05, seed=seed)

    model = nn.Network([nn.Linear(1, hidden), nn.Tanh(),
                        nn.Linear(hidden, 1)], nn.MSELoss())
    history = model.fit(X, Y, epochs=epochs, lr=lr, batch_size=batch_size,
                        verbose=True)
    print(f"\n  initial loss: {history[0]:.4f}")
    print(f"  final   loss: {history[-1]:.4f}")
    print()
    render_ascii_curve(model)


if __name__ == "__main__":
    main()
