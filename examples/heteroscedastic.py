"""Heteroscedastic regression demo: predict mean and input-dependent variance.

Synthetic 1-D function with state-dependent Gaussian noise:
  y = sin(x) + epsilon,  epsilon ~ N(0, sigma(x)^2),  sigma(x) = |x|/3 + 0.1.

Two networks of identical body but different heads:
  (a) MSE on a single scalar output: ordinary regression, no uncertainty.
  (b) Gaussian NLL on a two-output head (mu, z_s): per-input mu and
      sigma = softplus(z_s). Reports calibrated, input-dependent uncertainty.

Run: python examples/heteroscedastic.py
"""
import math
import random

import scratchnn as nn


def true_mean(x):
    return math.sin(x)


def true_std(x):
    return abs(x) / 3.0 + 0.1


def make_data(n, x_range=(0.0, 6.0), seed=0):
    rng = random.Random(seed)
    X = [[rng.uniform(*x_range)] for _ in range(n)]
    Y = [true_mean(x[0]) + rng.gauss(0.0, true_std(x[0])) for x in X]
    return X, Y


def main(n=400, epochs=300, lr=0.02, batch_size=20, seed=0):
    print("Generating heteroscedastic data...")
    X, Y = make_data(n, seed=seed)
    print(f"  {n} samples on x in [0, 6]; noise std = |x|/3 + 0.1 (grows with x)")

    # (a) Standard MSE regression.
    print(f"\n{'=' * 60}\n(a) MSE regression (homoscedastic assumption)\n{'=' * 60}")
    random.seed(seed)
    mse_model = nn.Network([
        nn.Linear(1, 16), nn.Tanh(),
        nn.Linear(16, 1),
    ], nn.MSELoss())
    Y_mse = [[y] for y in Y]
    mse_model.fit(X, Y_mse, epochs=epochs, lr=lr,
                  batch_size=batch_size, verbose=False)

    # (b) Heteroscedastic Gaussian NLL.
    print(f"\n{'=' * 60}\n(b) heteroscedastic Gaussian NLL (mu + sigma head)\n"
          f"{'=' * 60}")
    random.seed(seed)
    het_model = nn.Network([
        nn.Linear(1, 16), nn.Tanh(),
        nn.Linear(16, 2),
    ], nn.GaussianNLLLoss())
    het_model.fit(X, Y, epochs=epochs, lr=lr,
                  batch_size=batch_size, verbose=False)

    # Predictions at evenly spaced query points.
    print(f"\n{'x':>5} {'true mu':>9} {'true sd':>9} {'MSE pred':>10} "
          f"{'het mu':>9} {'het sd':>9}")
    print("-" * 58)
    for x in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        tm = true_mean(x)
        ts = true_std(x)
        mp = mse_model.predict([x])[0]
        hm, hs = het_model.predict([x])
        print(f"{x:>5.2f} {tm:>9.3f} {ts:>9.3f} {mp:>10.3f} "
              f"{hm:>9.3f} {hs:>9.3f}")

    # Held-out test NLL comparison.
    X_te, Y_te = make_data(200, seed=seed + 100)

    def gauss_nll(mu, sigma, y):
        return 0.5 * (y - mu) ** 2 / (sigma * sigma) + math.log(sigma)

    # For the MSE model, fit a global sigma post-hoc (root-mean-squared
    # residual) so we can score it with Gaussian NLL on equal footing.
    residuals = [(mse_model.predict([x[0]])[0] - y) ** 2 for x, y in zip(X, Y)]
    global_sigma = math.sqrt(sum(residuals) / len(residuals))
    print(f"\nMSE model's post-hoc global sigma (RMS train residual): "
          f"{global_sigma:.4f}")

    mse_nll = sum(
        gauss_nll(mse_model.predict([x[0]])[0], global_sigma, y)
        for x, y in zip(X_te, Y_te)
    ) / len(X_te)
    het_nll = sum(
        (lambda hm, hs: gauss_nll(hm, hs, y))(*het_model.predict([x[0]]))
        for x, y in zip(X_te, Y_te)
    ) / len(X_te)

    print(f"\nTest Gaussian NLL on {len(X_te)} held-out samples:")
    print(f"  MSE model + global sigma   : {mse_nll:.4f}")
    print(f"  Heteroscedastic head       : {het_nll:.4f}")
    print(f"  -> input-dependent sigma improves NLL by "
          f"{mse_nll - het_nll:.4f} nats per sample.")


if __name__ == "__main__":
    main()
