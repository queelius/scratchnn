"""Gradient checks: analytical gradients vs. central finite differences.

Run: python tests/test_gradients.py
"""
import random

import scratchnn as nn

TOL = 1e-4


def test_gradient_logistic_regression():
    random.seed(0)
    model = nn.Network([nn.Linear(3, 1)], nn.SigmoidBCE())
    error = nn.gradient_check(model, [0.5, -1.0, 2.0], 1)
    assert error < TOL, f"relative error {error}"


def test_gradient_softmax_regression():
    random.seed(1)
    model = nn.Network([nn.Linear(3, 4)], nn.SoftmaxCrossEntropy())
    error = nn.gradient_check(model, [0.5, -1.0, 2.0], 2)
    assert error < TOL, f"relative error {error}"


def test_gradient_tanh_mlp():
    random.seed(2)
    model = nn.Network([nn.Linear(3, 5), nn.Tanh(),
                        nn.Linear(5, 4)], nn.SoftmaxCrossEntropy())
    error = nn.gradient_check(model, [0.5, -1.0, 2.0], 0)
    assert error < TOL, f"relative error {error}"


def test_gradient_relu_mlp():
    random.seed(3)
    model = nn.Network([nn.Linear(3, 5), nn.ReLU(),
                        nn.Linear(5, 4)], nn.SoftmaxCrossEntropy())
    error = nn.gradient_check(model, [0.5, -1.0, 2.0], 3)
    # ReLU has a kink at 0; with random init and this input no
    # pre-activation lands on it, so the strict tolerance still holds.
    assert error < TOL, f"relative error {error}"


if __name__ == "__main__":
    import sys
    cases = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failed = 0
    for name, case in cases:
        try:
            case()
            print(f"PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)
