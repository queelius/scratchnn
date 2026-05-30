"""Recurrent layers for scratchnn.

A `RNNCell` is the stateful extension of the `Layer` protocol. Stateless
layers (`Linear`, `Tanh`, `ReLU`, `Conv2D`, ...) have
`forward(x) -> output` and `backward(g) -> dL/dx`. Stateful layers extend
this to
  `forward(x, state) -> (output, new_state)`
  `backward(g_out, dstate_next) -> (dL/dx, dL/dstate_prev)`
where the state is whatever the cell needs to remember between timesteps.
Both protocols share `parameters()`, but forward and backward shapes
diverge. This is the same Module/Functional split that real frameworks
navigate.

For a vanilla RNN cell the new state equals the output (both are
`h_new`), so the return tuple is `(h_new, h_new)`. For an LSTM the
output would be `h_new` while the state would be `(h_new, c_new)` (the
cell state is internal). The tuple convention generalizes.

Backward needs the values cached at forward time. The cell maintains an
internal cache stack: each `forward` pushes `(x, h_prev, h_new)`; each
`backward` pops the top and uses those values. The training loop is
responsible for ordering calls: forward T times then backward T times,
LIFO, with `reset_cache()` between sequences.
"""
import math
import random

from .neural_net import Layer


class RNNCell(Layer):
    """A vanilla recurrent cell with tanh activation:

        h_t = tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h).

    `forward(x, state)` returns `(h_new, h_new)`: the cell's output and
    its new state are the same vector. `backward(dh_out, dstate_next)`
    returns `(dx, dh_prev)`, summing the gradient that flowed back through
    the output projection (`dh_out`) and the gradient that flowed back
    through the next timestep (`dstate_next`) before applying the tanh
    derivative and accumulating weight gradients.

    Parameter storage matches `Linear`: one flat `list[float]` per row of
    `W_{xh}` and `W_{hh}` (i.e. per hidden unit), plus the bias vector.
    `parameters()` yields one (values, grads) pair per row, so `step()`
    and `zero_grad()` work unchanged.
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier-style init, scaled separately for the input and hidden
        # contributions. Both are linear maps into the hidden space.
        r_x = 1.0 / math.sqrt(input_size)
        r_h = 1.0 / math.sqrt(hidden_size)
        self.W_xh = [[random.uniform(-r_x, r_x) for _ in range(input_size)]
                     for _ in range(hidden_size)]
        self.W_hh = [[random.uniform(-r_h, r_h) for _ in range(hidden_size)]
                     for _ in range(hidden_size)]
        self.b_h = [0.0 for _ in range(hidden_size)]

        self.dW_xh = [[0.0 for _ in range(input_size)]
                      for _ in range(hidden_size)]
        self.dW_hh = [[0.0 for _ in range(hidden_size)]
                      for _ in range(hidden_size)]
        self.db_h = [0.0 for _ in range(hidden_size)]

        # LIFO cache of (x_t, h_{t-1}, h_t) tuples, one per forward call
        # since the last reset.
        self.cache = []

    def reset_cache(self):
        """Clear the per-timestep cache. Call between sequences."""
        self.cache = []

    def forward(self, x, state=None):
        H = self.hidden_size
        if state is None:
            state = [0.0 for _ in range(H)]
        h_new = [0.0] * H
        for i in range(H):
            z = self.b_h[i]
            w_xh_row = self.W_xh[i]
            for j in range(self.input_size):
                z += w_xh_row[j] * x[j]
            w_hh_row = self.W_hh[i]
            for j in range(H):
                z += w_hh_row[j] * state[j]
            h_new[i] = math.tanh(z)
        self.cache.append((x, state, h_new))
        return h_new, h_new

    def backward(self, dh_out, dstate_next=None):
        x, h_prev, h_new = self.cache.pop()
        H = self.hidden_size
        if dstate_next is None:
            dstate_next = [0.0 for _ in range(H)]

        # Total gradient on h_new: from this timestep's output, plus from
        # whatever propagated back from t+1 through h_new -> h_{t+1}.
        dh_total = [dh_out[i] + dstate_next[i] for i in range(H)]

        # Gradient through tanh: d/dz tanh(z) = 1 - tanh(z)^2 = 1 - h^2.
        da = [dh_total[i] * (1.0 - h_new[i] * h_new[i]) for i in range(H)]

        # Accumulate weight gradients. Outer products, same pattern as
        # Linear, just summed across the unrolled timesteps via repeated
        # backward calls.
        for i in range(H):
            dW_xh_row = self.dW_xh[i]
            for j in range(self.input_size):
                dW_xh_row[j] += da[i] * x[j]
            dW_hh_row = self.dW_hh[i]
            for j in range(H):
                dW_hh_row[j] += da[i] * h_prev[j]
            self.db_h[i] += da[i]

        # Backprop to input and to previous state.
        dx = [0.0 for _ in range(self.input_size)]
        for j in range(self.input_size):
            s = 0.0
            for i in range(H):
                s += self.W_xh[i][j] * da[i]
            dx[j] = s

        dh_prev = [0.0 for _ in range(H)]
        for j in range(H):
            s = 0.0
            for i in range(H):
                s += self.W_hh[i][j] * da[i]
            dh_prev[j] = s

        return dx, dh_prev

    def parameters(self):
        pairs = list(zip(self.W_xh, self.dW_xh))
        pairs += list(zip(self.W_hh, self.dW_hh))
        pairs.append((self.b_h, self.db_h))
        return pairs
