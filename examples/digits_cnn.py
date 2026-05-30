"""CNN demo on UCI 8x8 digits, with the permuted-pixel control.

Trains a small `Conv2D` + `Linear` network and reports test accuracy. Then
runs the headline ablation: with a fixed random permutation of the 64
input pixels, the MLP is unchanged (it had no spatial prior to lose)
while the CNN collapses (its locality prior is now actively wrong on the
data it sees).

Architectures:
  MLP : 64 -> 32 -> 10                                  ~2400 params
  CNN : Conv2D(1, 4, k=3) -> ReLU -> Linear(144, 10)    ~1500 params

Run: python examples/digits_cnn.py
"""
import os
import random
import sys

# Allow `import digits` when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scratchnn as nn
from digits import load_digits, TRAIN, TEST, accuracy


def build_cnn(seed):
    random.seed(seed)
    return nn.Network([
        nn.Conv2D(in_channels=1, out_channels=4, kernel_size=3,
                  in_h=8, in_w=8),
        nn.ReLU(),
        nn.Linear(4 * 6 * 6, 10),
    ], nn.SoftmaxCrossEntropy())


def build_mlp(hidden, seed):
    random.seed(seed)
    return nn.Network([
        nn.Linear(64, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 10),
    ], nn.SoftmaxCrossEntropy())


def permute_pixels(X, perm):
    """Apply a fixed permutation to the 64 input pixels."""
    return [[x[i] for i in perm] for x in X]


def count_params(model):
    return sum(len(values) for values, _ in model.parameters())


def fit_and_report(model, X_tr, Y_tr, X_te, Y_te, epochs, lr, batch_size,
                   label, verbose=True):
    print(f"\n{'-' * 60}\n{label}  ({count_params(model)} parameters)\n"
          f"{'-' * 60}")
    model.fit(X_tr, Y_tr, epochs=epochs, lr=lr, batch_size=batch_size,
              verbose=verbose)
    tr = accuracy(model, X_tr, Y_tr)
    te = accuracy(model, X_te, Y_te)
    print(f"  train acc: {tr:.4f}")
    print(f"  test  acc: {te:.4f}")
    return tr, te


def main(epochs=40, lr=0.1, batch_size=32, seed=0):
    print("Loading digits...")
    X_tr, Y_tr = load_digits(TRAIN)
    X_te, Y_te = load_digits(TEST)
    print(f"  train: {len(X_tr)} samples, test: {len(X_te)} samples")

    # Fixed random pixel permutation.
    rng = random.Random(seed + 100)
    perm = list(range(64))
    rng.shuffle(perm)
    Xp_tr = permute_pixels(X_tr, perm)
    Xp_te = permute_pixels(X_te, perm)

    results = {}

    cnn = build_cnn(seed)
    results["cnn_standard"] = fit_and_report(
        cnn, X_tr, Y_tr, X_te, Y_te,
        epochs=epochs, lr=lr, batch_size=batch_size,
        label="CNN on standard pixels", verbose=False)

    cnn_p = build_cnn(seed)
    results["cnn_permuted"] = fit_and_report(
        cnn_p, Xp_tr, Y_tr, Xp_te, Y_te,
        epochs=epochs, lr=lr, batch_size=batch_size,
        label="CNN on PERMUTED pixels (locality prior wrong)",
        verbose=False)

    mlp = build_mlp(32, seed)
    results["mlp_standard"] = fit_and_report(
        mlp, X_tr, Y_tr, X_te, Y_te,
        epochs=epochs, lr=lr, batch_size=batch_size,
        label="MLP on standard pixels", verbose=False)

    mlp_p = build_mlp(32, seed)
    results["mlp_permuted"] = fit_and_report(
        mlp_p, Xp_tr, Y_tr, Xp_te, Y_te,
        epochs=epochs, lr=lr, batch_size=batch_size,
        label="MLP on PERMUTED pixels (no prior to lose)",
        verbose=False)

    # Summary table.
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"{'model':<20} {'pixels':<12} {'params':>8} {'test acc':>10}")
    print(f"{'-' * 52}")
    rows = [
        ("MLP (64-32-10)", "standard", count_params(mlp),
         results["mlp_standard"][1]),
        ("MLP (64-32-10)", "permuted", count_params(mlp_p),
         results["mlp_permuted"][1]),
        ("CNN (1,4,k=3+L)", "standard", count_params(cnn),
         results["cnn_standard"][1]),
        ("CNN (1,4,k=3+L)", "permuted", count_params(cnn_p),
         results["cnn_permuted"][1]),
    ]
    for name, pixels, p, acc in rows:
        print(f"{name:<20} {pixels:<12} {p:>8} {acc:>9.4f}")

    cnn_drop = results["cnn_standard"][1] - results["cnn_permuted"][1]
    mlp_drop = results["mlp_standard"][1] - results["mlp_permuted"][1]
    print(f"\nCNN test-accuracy drop under pixel permutation: {cnn_drop:+.4f}")
    print(f"MLP test-accuracy drop under pixel permutation: {mlp_drop:+.4f}")


if __name__ == "__main__":
    main()
