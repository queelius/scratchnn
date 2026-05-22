"""Unit tests for scratchnn. Run: python tests/test_neural_net.py"""
import math
import random

import scratchnn as nn


def approx(a, b, tol=1e-9):
    """True if scalars a and b are within tol of each other."""
    return abs(a - b) <= tol


# ---- tests ---------------------------------------------------------------

def test_dot():
    assert approx(nn.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)
    assert approx(nn.dot([], []), 0.0)


def test_sigmoid_midpoint():
    assert approx(nn.sigmoid(0.0), 0.5)


def test_sigmoid_symmetry():
    assert approx(nn.sigmoid(2.0) + nn.sigmoid(-2.0), 1.0)


def test_sigmoid_no_overflow():
    assert approx(nn.sigmoid(1000.0), 1.0)
    assert approx(nn.sigmoid(-1000.0), 0.0)


def test_softmax_sums_to_one():
    assert approx(sum(nn.softmax([1.0, 2.0, 3.0])), 1.0)


def test_softmax_shift_invariance():
    a = nn.softmax([1.0, 2.0, 3.0])
    b = nn.softmax([101.0, 102.0, 103.0])
    for ai, bi in zip(a, b):
        assert approx(ai, bi)


def test_softmax_no_overflow():
    for pi in nn.softmax([1000.0, 1000.0, 1000.0]):
        assert approx(pi, 1.0 / 3.0)


def test_logsumexp_matches_naive():
    zs = [0.1, 0.2, 0.3]
    naive = math.log(sum(math.exp(z) for z in zs))
    assert approx(nn.logsumexp(zs), naive)


def test_logsumexp_no_overflow():
    assert approx(nn.logsumexp([1000.0, 1000.0]), 1000.0 + math.log(2.0))


def test_linear_forward():
    layer = nn.Linear(2, 2)
    layer.weights = [[1.0, 2.0], [3.0, 4.0]]
    layer.bias = [0.5, -0.5]
    out = layer.forward([1.0, 1.0])
    assert approx(out[0], 3.5)
    assert approx(out[1], 6.5)


def test_linear_backward_input_grad():
    layer = nn.Linear(2, 2)
    layer.weights = [[1.0, 2.0], [3.0, 4.0]]
    layer.bias = [0.0, 0.0]
    layer.forward([1.0, 1.0])
    dx = layer.backward([1.0, 1.0])
    assert approx(dx[0], 4.0)
    assert approx(dx[1], 6.0)


def test_linear_backward_param_grad():
    layer = nn.Linear(2, 1)
    layer.weights = [[0.0, 0.0]]
    layer.bias = [0.0]
    layer.forward([2.0, 3.0])
    layer.backward([5.0])
    assert approx(layer.dweights[0][0], 10.0)
    assert approx(layer.dweights[0][1], 15.0)
    assert approx(layer.dbias[0], 5.0)


def test_linear_backward_accumulates():
    layer = nn.Linear(2, 1)
    layer.weights = [[0.0, 0.0]]
    layer.bias = [0.0]
    layer.forward([1.0, 1.0])
    layer.backward([1.0])
    layer.forward([1.0, 1.0])
    layer.backward([1.0])
    assert approx(layer.dbias[0], 2.0)


def test_linear_init_shape_and_bias():
    layer = nn.Linear(3, 4)
    assert len(layer.weights) == 4
    assert all(len(w) == 3 for w in layer.weights)
    assert all(b == 0.0 for b in layer.bias)


def test_linear_parameters_pairs():
    params = nn.Linear(2, 2).parameters()
    assert len(params) == 3
    for values, grads in params:
        assert len(values) == len(grads)


def test_tanh_forward():
    assert approx(nn.Tanh().forward([0.0])[0], 0.0)


def test_tanh_backward():
    layer = nn.Tanh()
    layer.forward([0.0])
    # derivative at 0 is 1 - tanh(0)**2 = 1, so g passes through unchanged
    assert approx(layer.backward([2.0])[0], 2.0)


def test_relu_forward():
    assert nn.ReLU().forward([-3.0, 0.0, 4.0]) == [0.0, 0.0, 4.0]


def test_relu_backward():
    layer = nn.ReLU()
    layer.forward([-3.0, 0.0, 4.0])
    assert layer.backward([1.0, 1.0, 1.0]) == [0.0, 0.0, 1.0]


def test_sigmoid_bce_value():
    loss = nn.SigmoidBCE()
    # at logit 0, p = 0.5, so BCE = -log(0.5) = log(2) for either label
    assert approx(loss.value([0.0], 1), math.log(2.0))
    assert approx(loss.value([0.0], 0), math.log(2.0))


def test_sigmoid_bce_grad():
    loss = nn.SigmoidBCE()
    # grad = sigmoid(z) - y; at z = 0, sigmoid = 0.5
    assert approx(loss.grad([0.0], 1)[0], -0.5)
    assert approx(loss.grad([0.0], 0)[0], 0.5)


def test_sigmoid_bce_probs():
    assert approx(nn.SigmoidBCE().probs([0.0])[0], 0.5)


def test_softmax_ce_value():
    # uniform logits over 3 classes -> p = 1/3 -> loss = -log(1/3) = log(3)
    assert approx(nn.SoftmaxCrossEntropy().value([0.0, 0.0, 0.0], 1),
                  math.log(3.0))


def test_softmax_ce_grad():
    g = nn.SoftmaxCrossEntropy().grad([0.0, 0.0, 0.0], 1)
    # p - onehot(1) = [1/3, 1/3 - 1, 1/3]
    assert approx(g[0], 1.0 / 3.0)
    assert approx(g[1], 1.0 / 3.0 - 1.0)
    assert approx(g[2], 1.0 / 3.0)


def test_softmax_ce_probs():
    p = nn.SoftmaxCrossEntropy().probs([0.0, 0.0, 0.0])
    assert approx(sum(p), 1.0)
    assert approx(p[0], 1.0 / 3.0)


def test_network_forward_is_layer_composition():
    layer = nn.Linear(2, 1)
    layer.weights = [[1.0, 1.0]]
    layer.bias = [0.0]
    model = nn.Network([layer], nn.SigmoidBCE())
    assert approx(model.forward([2.0, 3.0])[0], 5.0)


def test_network_predict_returns_probabilities():
    layer = nn.Linear(2, 1)
    layer.weights = [[0.0, 0.0]]
    layer.bias = [0.0]
    model = nn.Network([layer], nn.SigmoidBCE())
    assert approx(model.predict([1.0, 1.0])[0], 0.5)


def test_network_loss_value():
    layer = nn.Linear(2, 1)
    layer.weights = [[0.0, 0.0]]
    layer.bias = [0.0]
    model = nn.Network([layer], nn.SigmoidBCE())
    assert approx(model.loss_value([1.0, 1.0], 1), math.log(2.0))


def test_network_zero_grad():
    model = nn.Network([nn.Linear(2, 1)], nn.SigmoidBCE())
    model.forward([1.0, 1.0])
    model.backward(1)
    model.zero_grad()
    for _values, grads in model.parameters():
        assert all(g == 0.0 for g in grads)


def test_network_step_updates_weights():
    layer = nn.Linear(1, 1)
    layer.weights = [[2.0]]
    layer.bias = [0.0]
    layer.dweights = [[4.0]]
    layer.dbias = [0.0]
    model = nn.Network([layer], nn.SigmoidBCE())
    model.step(lr=1.0, n=2)
    # w -= (lr / n) * g = 2.0 - (1/2) * 4.0 = 0.0
    assert approx(layer.weights[0][0], 0.0)


def test_mlp_forward_runs():
    model = nn.Network([nn.Linear(2, 4), nn.Tanh(),
                        nn.Linear(4, 3)], nn.SoftmaxCrossEntropy())
    p = model.predict([0.5, -0.5])
    assert len(p) == 3
    assert approx(sum(p), 1.0)


def test_network_fit_reduces_loss():
    random.seed(0)
    # classes are linearly separable by the sign of the first feature
    X = [[-2.0, 0.3], [-1.0, -0.4], [1.0, 0.5], [2.0, -0.2]]
    Y = [0, 0, 1, 1]
    model = nn.Network([nn.Linear(2, 1)], nn.SigmoidBCE())
    history = model.fit(X, Y, epochs=100, lr=0.5, batch_size=2)
    assert len(history) == 100
    assert history[-1] < history[0]
    assert history[-1] < 0.3


def test_network_fit_invokes_callback():
    random.seed(0)
    seen = []

    def record(epoch, net, history):
        seen.append((epoch, len(history)))

    model = nn.Network([nn.Linear(2, 1)], nn.SigmoidBCE())
    model.fit([[1.0, 0.0], [0.0, 1.0]], [0, 1], epochs=3, lr=0.1,
              callback=record)
    assert [e for e, _ in seen] == [0, 1, 2]
    assert [h for _, h in seen] == [1, 2, 3]


# ---- runner --------------------------------------------------------------

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
