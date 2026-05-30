"""Poisson regression demo: count data with a log-link Poisson NLL.

A synthetic 1-D regression where the conditional mean (rate) is non-monotonic
and the targets are Poisson counts. Compares:

  (a) identity link + MSE: ordinary regression. The model can predict
      *negative* rates, which are impossible for counts, and it assumes
      constant variance.
  (b) log link + Poisson NLL: predicts log-rate; the rate lambda = exp(z)
      is guaranteed positive by construction. Accounts for Var = Mean of
      the Poisson.

Run: python examples/poisson_regression.py
"""
import math
import random

import scratchnn as nn


def true_rate(x):
    """Non-monotonic positive rate as a function of x in [0, 2]."""
    return max(0.1, 2.0 + 5.0 * math.sin(math.pi * x))


def poisson_sample(lam, rng):
    """Sample one Poisson variate via Knuth's method (small-lambda regime)."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def make_data(n, seed=0):
    rng = random.Random(seed)
    X = [[rng.uniform(0.0, 2.0)] for _ in range(n)]
    Y = [poisson_sample(true_rate(x[0]), rng) for x in X]
    return X, Y


def main(n_train=500, n_test=200, epochs=80, lr=0.05, batch_size=32, seed=0):
    print("Generating Poisson count data...")
    X_tr, Y_tr = make_data(n_train, seed=seed)
    X_te, Y_te = make_data(n_test, seed=seed + 100)
    print(f"  {n_train} training samples, {n_test} test samples")
    print(f"  train y range: [{min(Y_tr)}, {max(Y_tr)}]")

    # (a) identity + MSE on counts.
    print(f"\n{'=' * 60}\n(a) identity link + MSE on count data\n{'=' * 60}")
    random.seed(seed)
    mse_model = nn.Network([
        nn.Linear(1, 16), nn.Tanh(),
        nn.Linear(16, 1),
    ], nn.MSELoss())
    Y_tr_mse = [[float(y)] for y in Y_tr]
    mse_model.fit(X_tr, Y_tr_mse, epochs=epochs, lr=lr,
                  batch_size=batch_size, verbose=False)

    # (b) log + Poisson NLL on counts.
    print(f"\n{'=' * 60}\n(b) log link + Poisson NLL on count data\n{'=' * 60}")
    random.seed(seed)
    pois_model = nn.Network([
        nn.Linear(1, 16), nn.Tanh(),
        nn.Linear(16, 1),
    ], nn.PoissonNLLLoss())
    pois_model.fit(X_tr, Y_tr, epochs=epochs, lr=lr,
                   batch_size=batch_size, verbose=False)

    # Predictions on a fine grid.
    grid = [i * 2.0 / 50 for i in range(51)]
    print(f"\n{'x':>6} {'true rate':>10} {'MSE pred':>10} {'Poisson pred':>14}")
    print("-" * 44)
    for x in grid[::5]:
        tr = true_rate(x)
        mp = mse_model.predict([x])[0]
        pp = pois_model.predict([x])[0]
        print(f"{x:>6.2f} {tr:>10.3f} {mp:>10.3f} {pp:>14.3f}")

    # Count negative predictions across the full grid (load-bearing point).
    mse_negs = sum(1 for x in grid if mse_model.predict([x])[0] < 0)
    pois_negs = sum(1 for x in grid if pois_model.predict([x])[0] < 0)
    print(f"\nNegative-rate predictions on a {len(grid)}-point grid in [0, 2]:")
    print(f"  MSE model     : {mse_negs}  (impossible for counts)")
    print(f"  Poisson model : {pois_negs}  (guaranteed positive by exp link)")

    # Test loss under each model's own loss.
    def mean_loss(model, X, Y):
        total = 0.0
        for x, y in zip(X, Y):
            total += model.loss.value(model.forward(x), y)
        return total / len(X)

    Y_te_mse = [[float(y)] for y in Y_te]
    mse_test = mean_loss(mse_model, X_te, Y_te_mse)
    pois_test = mean_loss(pois_model, X_te, Y_te)

    # Apples-to-apples: compute Poisson NLL of each model's *rate prediction*
    # on the test set. For MSE we clip predicted rate at 1e-3 to avoid log(0).
    def poisson_nll_at(pred_rate, y):
        rate = max(pred_rate, 1e-3)
        return rate - y * math.log(rate)

    mse_pois_nll = sum(
        poisson_nll_at(mse_model.predict([x[0]])[0], y)
        for x, y in zip(X_te, Y_te)
    ) / len(X_te)
    pois_pois_nll = sum(
        poisson_nll_at(pois_model.predict([x[0]])[0], y)
        for x, y in zip(X_te, Y_te)
    ) / len(X_te)

    print(f"\nTest metrics on {n_test} held-out samples:")
    print(f"  MSE model        own MSE loss     : {mse_test:.4f}")
    print(f"  Poisson model    own Poisson NLL  : {pois_test:.4f}")
    print(f"  MSE model        Poisson NLL      : {mse_pois_nll:.4f}")
    print(f"  Poisson model    Poisson NLL      : {pois_pois_nll:.4f}")
    print(f"  -> Poisson head improves Poisson NLL by "
          f"{mse_pois_nll - pois_pois_nll:.4f} nats per sample.")


if __name__ == "__main__":
    main()
