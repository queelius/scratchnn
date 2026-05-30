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


def test_gradient_mse_scalar():
    random.seed(4)
    model = nn.Network([nn.Linear(3, 1)], nn.MSELoss())
    error = nn.gradient_check(model, [0.5, -1.0, 2.0], [0.7])
    assert error < TOL, f"relative error {error}"


def test_gradient_mse_mlp():
    random.seed(5)
    model = nn.Network([nn.Linear(3, 5), nn.Tanh(),
                        nn.Linear(5, 2)], nn.MSELoss())
    error = nn.gradient_check(model, [0.5, -1.0, 2.0], [0.3, -0.7])
    assert error < TOL, f"relative error {error}"


def test_gradient_conv2d_mse():
    # Conv2D alone with MSE: 1 in channel, 4x4 input, 2 out channels, k=3.
    # Output is 2 * 2 * 2 = 8 floats.
    random.seed(6)
    model = nn.Network([nn.Conv2D(1, 2, 3, 4, 4)], nn.MSELoss())
    x = [0.5, -1.0, 2.0, 0.3, -0.7, 1.1, 0.0, -0.4,
         0.9, -0.2, 0.8, 1.5, -1.0, 0.3, 0.6, -0.5]
    y = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    error = nn.gradient_check(model, x, y)
    assert error < TOL, f"relative error {error}"


def test_gradient_conv2d_softmax_mlp():
    # Conv2D -> Tanh -> Linear -> softmax: a small classification head.
    # 1 in channel, 5x5 input, 2 out channels, k=3 -> 2*3*3 = 18 features.
    random.seed(7)
    conv = nn.Conv2D(1, 2, 3, 5, 5)
    model = nn.Network([conv, nn.Tanh(),
                        nn.Linear(conv.out_size, 3)],
                       nn.SoftmaxCrossEntropy())
    rng = random.Random(99)
    x = [rng.uniform(-1.0, 1.0) for _ in range(25)]
    error = nn.gradient_check(model, x, 1)
    assert error < TOL, f"relative error {error}"


def test_gradient_conv_pool_softmax():
    # Conv2D -> ReLU -> GlobalAvgPool -> Linear -> softmax: the
    # translation-invariant classifier used in the synthetic-shapes demo.
    random.seed(8)
    conv = nn.Conv2D(1, 3, 3, 5, 5)
    pool = nn.GlobalAvgPool(3, conv.out_h, conv.out_w)
    model = nn.Network([conv, nn.Tanh(), pool,
                        nn.Linear(pool.out_size, 4)],
                       nn.SoftmaxCrossEntropy())
    rng = random.Random(101)
    x = [rng.uniform(-1.0, 1.0) for _ in range(25)]
    error = nn.gradient_check(model, x, 2)
    assert error < TOL, f"relative error {error}"


def test_gradient_embed_concat_network():
    """Verify the EmbedConcat wrapper composes correctly in a Network.

    Mirrors the Bengio 2003 architecture as a vanilla `Network`:
    EmbedConcat -> Linear -> Tanh -> Linear -> SoftmaxCrossEntropy. The
    standard `gradient_check` then handles the rest.
    """
    random.seed(11)
    VOCAB, CONTEXT, EMBED, HIDDEN = 5, 3, 2, 4
    model = nn.Network([
        nn.EmbedConcat(VOCAB, EMBED, CONTEXT),
        nn.Linear(CONTEXT * EMBED, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, VOCAB),
    ], nn.SoftmaxCrossEntropy())
    rng = random.Random(789)
    x = [rng.randint(0, VOCAB - 1) for _ in range(CONTEXT)]
    y = rng.randint(0, VOCAB - 1)
    error = nn.gradient_check(model, x, y)
    assert error < TOL, f"relative error {error}"


def test_gradient_embedding_fixed_context():
    """Verify gradients through an Embedding + concat + MLP head.

    Mirrors the Bengio 2003 fixed-context LM: look up N=3 embeddings of
    size EMBED=2, concatenate to 6 dims, run through Linear+Tanh+Linear
    to vocab logits, softmax-CE loss. Check every parameter against
    finite differences.
    """
    random.seed(10)
    VOCAB, CONTEXT, EMBED, HIDDEN = 5, 3, 2, 4
    EPS = 1e-5

    emb = nn.Embedding(VOCAB, EMBED)
    fc1 = nn.Linear(CONTEXT * EMBED, HIDDEN)
    act = nn.Tanh()
    fc2 = nn.Linear(HIDDEN, VOCAB)
    loss = nn.SoftmaxCrossEntropy()

    rng = random.Random(456)
    context_ids = [rng.randint(0, VOCAB - 1) for _ in range(CONTEXT)]
    target = rng.randint(0, VOCAB - 1)

    def total_loss():
        emb.reset_cache()
        x = []
        for c in context_ids:
            x.extend(emb.forward(c))
        h = fc1.forward(x)
        h = act.forward(h)
        logits = fc2.forward(h)
        return loss.value(logits, target)

    # Zero all gradient accumulators.
    for row in emb.dweights:
        for k in range(len(row)):
            row[k] = 0.0
    for row in fc1.dweights:
        for k in range(len(row)):
            row[k] = 0.0
    for k in range(len(fc1.dbias)):
        fc1.dbias[k] = 0.0
    for row in fc2.dweights:
        for k in range(len(row)):
            row[k] = 0.0
    for k in range(len(fc2.dbias)):
        fc2.dbias[k] = 0.0

    # Analytical forward and backward.
    emb.reset_cache()
    x = []
    for c in context_ids:
        x.extend(emb.forward(c))
    h_pre = fc1.forward(x)
    h_post = act.forward(h_pre)
    logits = fc2.forward(h_post)

    d_logits = loss.grad(logits, target)
    d_h_post = fc2.backward(d_logits)
    d_h_pre = act.backward(d_h_post)
    d_x = fc1.backward(d_h_pre)
    # Split d_x and backprop to embeddings in LIFO order to match cache.
    for p in range(CONTEXT - 1, -1, -1):
        emb.backward(d_x[p * EMBED:(p + 1) * EMBED])

    def check(values, grads):
        worst_here = 0.0
        for k in range(len(values)):
            original = values[k]
            values[k] = original + EPS
            l_plus = total_loss()
            values[k] = original - EPS
            l_minus = total_loss()
            values[k] = original
            numerical = (l_plus - l_minus) / (2 * EPS)
            denom = max(abs(numerical) + abs(grads[k]), 1e-12)
            err = abs(numerical - grads[k]) / denom
            if err > worst_here:
                worst_here = err
        return worst_here

    worst = 0.0
    for row, drow in zip(emb.weights, emb.dweights):
        worst = max(worst, check(row, drow))
    for row, drow in zip(fc1.weights, fc1.dweights):
        worst = max(worst, check(row, drow))
    worst = max(worst, check(fc1.bias, fc1.dbias))
    for row, drow in zip(fc2.weights, fc2.dweights):
        worst = max(worst, check(row, drow))
    worst = max(worst, check(fc2.bias, fc2.dbias))
    assert worst < TOL, f"worst Embedding+MLP relative error {worst}"


def test_gradient_rnn_unrolled_bptt():
    """Verify backprop-through-time matches finite differences.

    Unroll an RNNCell for T timesteps with a Linear output projection,
    sum softmax-cross-entropy losses across timesteps, and check that
    every parameter's analytical gradient matches a central-difference
    numerical gradient. Custom check because the standard
    `gradient_check` does not handle stateful unrolling.
    """
    random.seed(9)
    INPUT_SIZE, HIDDEN_SIZE, VOCAB, T = 3, 4, 5, 3
    EPS = 1e-5

    cell = nn.RNNCell(INPUT_SIZE, HIDDEN_SIZE)
    proj = nn.Linear(HIDDEN_SIZE, VOCAB)
    loss = nn.SoftmaxCrossEntropy()

    rng = random.Random(123)
    xs = [[rng.uniform(-1, 1) for _ in range(INPUT_SIZE)] for _ in range(T)]
    ys = [rng.randint(0, VOCAB - 1) for _ in range(T)]

    def total_loss():
        cell.reset_cache()
        h_state = None
        L = 0.0
        for t in range(T):
            h_out, h_state = cell.forward(xs[t], h_state)
            logits = proj.forward(h_out)
            L += loss.value(logits, ys[t])
        return L

    # Analytical gradients via BPTT.
    cell.reset_cache()
    for row in cell.dW_xh:
        for k in range(len(row)):
            row[k] = 0.0
    for row in cell.dW_hh:
        for k in range(len(row)):
            row[k] = 0.0
    for k in range(len(cell.db_h)):
        cell.db_h[k] = 0.0
    for row in proj.dweights:
        for k in range(len(row)):
            row[k] = 0.0
    for k in range(len(proj.dbias)):
        proj.dbias[k] = 0.0

    h_state = None
    h_outs = []
    logits_list = []
    for t in range(T):
        h_out, h_state = cell.forward(xs[t], h_state)
        h_outs.append(h_out)
        logits_list.append(proj.forward(h_out))

    dh_next = [0.0] * HIDDEN_SIZE
    for t in range(T - 1, -1, -1):
        # Re-run proj.forward to refresh proj.x cache for this timestep
        # (proj overwrites .x on each forward).
        proj.forward(h_outs[t])
        d_logits = loss.grad(logits_list[t], ys[t])
        dh_from_proj = proj.backward(d_logits)
        _, dh_next = cell.backward(dh_from_proj, dh_next)

    def check(values, grads):
        worst_here = 0.0
        for k in range(len(values)):
            original = values[k]
            values[k] = original + EPS
            l_plus = total_loss()
            values[k] = original - EPS
            l_minus = total_loss()
            values[k] = original
            numerical = (l_plus - l_minus) / (2 * EPS)
            denom = max(abs(numerical) + abs(grads[k]), 1e-12)
            err = abs(numerical - grads[k]) / denom
            if err > worst_here:
                worst_here = err
        return worst_here

    worst = 0.0
    for row, drow in zip(cell.W_xh, cell.dW_xh):
        worst = max(worst, check(row, drow))
    for row, drow in zip(cell.W_hh, cell.dW_hh):
        worst = max(worst, check(row, drow))
    worst = max(worst, check(cell.b_h, cell.db_h))
    for row, drow in zip(proj.weights, proj.dweights):
        worst = max(worst, check(row, drow))
    worst = max(worst, check(proj.bias, proj.dbias))

    assert worst < TOL, f"worst RNN BPTT relative error {worst}"


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
