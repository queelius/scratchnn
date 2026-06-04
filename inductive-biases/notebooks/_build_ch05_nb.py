"""Construct notebooks/ch05-rnn.ipynb via nbformat.

Run this, then execute the notebook with nbconvert so every cell carries a
stored output. Kept in the repo so the notebook is reproducible from a
structured source rather than hand-edited JSON. Mirrors the Chapter 4
builder (_build_ch04_nb.py): same conventions, same Alice corpus and
vocabulary, so the recurrent network and the fixed-context model are
trained on identical data.

Ported from examples/text_rnn.py: the BPTT training loop (forward T times,
backward T times LIFO, global grad-norm clip, mean-gradient step) lives in
the notebook so a reader sees the whole experiment in one place.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(text):
    cells.append(new_code_cell(text))


md(r"""# Chapter 5: Recurrent Networks

Paired notebook for Chapter 5 of *Inductive Biases in Neural Networks*.
Trains a vanilla recurrent network on a char-level subset of *Alice's
Adventures in Wonderland*: a stateful `RNNCell` threads a hidden state
forward one character at a time, and a `Linear` head projects the state to
logits over the vocabulary. Backpropagation through time (BPTT) unrolls the
cell over short windows and accumulates the recurrent weight gradients
across timesteps. `SoftmaxCrossEntropy` is the output head. Pure-Python
`scratchnn` throughout, model and experiment.

**Same data as Chapter 4.** The corpus is the first 30,000 characters of
Alice, a 75-character vocabulary, identical to the fixed-context language
model. That is deliberate: the two models are set against each other on the
same data, so the comparison at the end is fair.

**Seeds:** `random.seed(0)` before the cell and head are built (this drives
weight initialization and the per-epoch chunk shuffle); each per-epoch
sample re-seeds `random.seed(1234 + epoch)` so the quoted samples are
reproducible. The gradient check uses `random.seed(7)`; the
vanishing-gradient demonstration uses `random.seed(0)`.

**Figure saved:** `../book/figures/ch05-rnn-loss.pdf`

**Headline numbers (regenerated below):** 30,000 chars, 75-char vocab, 937
BPTT chunks of length 33; `RNNCell(75, 64) -> Linear(64, 75)`; 13,835
parameters; sequence length 32, learning rate 0.5, global grad-norm clip 5,
15 epochs. Mean per-character loss falls from the uniform baseline
`log 75 ~ 4.32` nats toward roughly **2.06** nats/char over the 15 epochs.
`gradient_check` on the unrolled cell (T=3) is on the order of 1e-9. The
vanishing-gradient demonstration shows the backward Jacobian-product norm
decaying over unroll depth.""")

md(r"""## Setup

The library is pure Python (`math`, `random` only). The corpus loader and
vocabulary builder come from `examples/text_rnn.py`, which reads
`examples/data/alice.txt` (Project Gutenberg eBook #11, header and footer
stripped) and takes the first 30,000 characters. This is the same loader
and the same 30,000-character slice Chapter 4 used.""")

code(r"""import os
import sys
import math
import random

# Pure-Python scratchnn plus the shared Alice loader from examples/.
EXAMPLES = "/home/spinoza/github/repos/scratchnn/examples"
sys.path.insert(0, EXAMPLES)

import scratchnn as nn
from text_rnn import load_alice, build_vocab

text = load_alice(max_chars=30000)
chars, c2i, i2c = build_vocab(text)
vocab_size = len(chars)
ids = [c2i[c] for c in text]

print(f"corpus: {len(text)} chars; vocab: {vocab_size} chars")
print(f"uniform-random baseline: log(vocab) = {math.log(vocab_size):.4f} nats/char")""")

md(r"""## The model

Two pieces. `RNNCell(vocab, hidden)` is the stateful recurrent cell

$$h_t = \tanh\!\bigl(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h\bigr),$$

and `Linear(hidden, vocab)` projects the hidden state to logits. The cell
runs the same computation at every step; only the state and the input
change. The parameter count is the input-to-hidden matrix, the
hidden-to-hidden matrix, the hidden bias, and the output projection with
its bias:

$$64 \cdot 75 \;+\; 64 \cdot 64 \;+\; 64 \;+\; 75 \cdot 64 \;+\; 75 \;=\; 13{,}835.$$""")

code(r"""HIDDEN = 64


def build_rnn(seed):
    random.seed(seed)
    cell = nn.RNNCell(vocab_size, HIDDEN)
    proj = nn.Linear(HIDDEN, vocab_size)
    return cell, proj


_cell, _proj = build_rnn(0)
n_params = (sum(len(v) for v, _ in _cell.parameters())
            + sum(len(v) for v, _ in _proj.parameters()))
hand_count = (HIDDEN * vocab_size + HIDDEN * HIDDEN + HIDDEN
              + vocab_size * HIDDEN + vocab_size)
print(f"parameters (from layers): {n_params}")
print(f"parameters (hand count):  {hand_count}")""")

md(r"""## Training data: BPTT chunks

The corpus is cut into non-overlapping windows of length `seq_len + 1`, so
each window gives `seq_len` input/target pairs: at step `t` the model reads
character `t` and is asked to predict character `t + 1`. Unlike the
fixed-context model, the chunk is processed *sequentially*, threading the
hidden state from one step to the next, and the gradient is propagated back
through the whole chunk.""")

code(r"""SEQ_LEN = 32

chunks = [ids[i:i + SEQ_LEN + 1]
          for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN)]
print(f"chunks: {len(chunks)} of length {SEQ_LEN + 1} "
      f"({SEQ_LEN} input/target pairs each)")""")

md(r"""## One hot

The cell takes a real vector as input. A character id becomes a one-hot
vector of length `vocab_size`; the input-to-hidden matrix then selects the
matching column, exactly as an `Embedding` lookup would, kept explicit
here.""")

code(r"""def one_hot(idx, n):
    v = [0.0] * n
    v[idx] = 1.0
    return v""")

md(r"""## The BPTT training step

This is the heart of the experiment, ported from `examples/text_rnn.py`.
One chunk is one SGD step:

1. **Forward** through the chunk, threading the state `h_0 -> h_1 -> ... ->
   h_T`. At each step the `Linear` head produces logits and the loss adds a
   per-step term. The total loss is the sum over timesteps.
2. **Backward** in reverse (LIFO), so the cell's cache pops the matching
   timestep. Each cell `backward` receives two gradients: `dh_out` from this
   step's output projection, and `dstate_next` from the future through
   `h_{t+1}`. The cell adds them, applies the tanh derivative, and
   accumulates the recurrent weight gradients with `+=`.
3. **Clip** the global gradient L2 norm to 5 (the standard mitigation for
   exploding gradients in vanilla RNNs).
4. **Step** against the *mean* per-timestep gradient: the gradients summed
   across `T` timesteps are divided by `T`, the same convention
   `Network.step(lr, n)` uses for a mini-batch.""")

code(r"""def zero_grads(*layers):
    for layer in layers:
        for _, g in layer.parameters():
            for k in range(len(g)):
                g[k] = 0.0


def global_grad_norm(*layers):
    total = 0.0
    for layer in layers:
        for _, g in layer.parameters():
            for gk in g:
                total += gk * gk
    return math.sqrt(total)


def clip_grads(*layers, max_norm=5.0):
    norm = global_grad_norm(*layers)
    if norm > max_norm:
        scale = max_norm / norm
        for layer in layers:
            for _, g in layer.parameters():
                for k in range(len(g)):
                    g[k] *= scale
    return norm


def sgd_step(lr, n, *layers):
    # Mean accumulated gradient: divide the summed gradient by n (= T).
    factor = lr / n
    for layer in layers:
        for v, g in layer.parameters():
            for k in range(len(v)):
                v[k] -= factor * g[k]


def train_one_chunk(cell, proj, loss, chunk_ids, lr):
    T = len(chunk_ids) - 1
    zero_grads(cell, proj)
    cell.reset_cache()

    # Forward: thread the state, collect per-step logits and targets.
    h_outs, logits_list, targets = [], [], []
    h_state = None
    L = 0.0
    for t in range(T):
        x = one_hot(chunk_ids[t], vocab_size)
        h_out, h_state = cell.forward(x, h_state)
        h_outs.append(h_out)
        logits = proj.forward(h_out)
        logits_list.append(logits)
        targets.append(chunk_ids[t + 1])
        L += loss.value(logits, chunk_ids[t + 1])

    # Backward through time, LIFO. dh_next carries the future gradient.
    dh_next = [0.0] * cell.hidden_size
    for t in range(T - 1, -1, -1):
        proj.forward(h_outs[t])             # refresh proj's cache for step t
        d_logits = loss.grad(logits_list[t], targets[t])
        dh_from_proj = proj.backward(d_logits)
        _, dh_next = cell.backward(dh_from_proj, dh_next)

    clip_grads(cell, proj, max_norm=5.0)
    sgd_step(lr, T, cell, proj)
    return L / T""")

md(r"""## Gradient check on the unrolled cell

BPTT is only worth running if it is correct. Unroll the cell for `T = 3`
with a small `Linear` output projection and `SoftmaxCrossEntropy`, build the
analytic gradient with one forward/backward sweep, and compare every
parameter against a central finite difference. The check perturbs each
parameter in the cell and the head, recomputes the summed loss over the
three steps, and reports the worst relative error. It should sit near
machine precision, far under the `1e-4` tolerance. (`scratchnn`'s
`tests/test_gradients.py::test_gradient_rnn_unrolled_bptt` runs the same
check.)""")

code(r"""def unrolled_loss(cell, proj, loss, xs, ys):
    "Summed loss over an unrolled sequence (forward only)."
    cell.reset_cache()
    h_state = None
    L = 0.0
    for x, y in zip(xs, ys):
        h_out, h_state = cell.forward(x, h_state)
        logits = proj.forward(h_out)
        L += loss.value(logits, y)
    return L


def unrolled_grads(cell, proj, loss, xs, ys):
    "Analytic gradient of the summed loss via BPTT."
    zero_grads(cell, proj)
    cell.reset_cache()
    h_outs, logits_list = [], []
    h_state = None
    for x in xs:
        h_out, h_state = cell.forward(x, h_state)
        h_outs.append(h_out)
        logits_list.append(proj.forward(h_out))
    T = len(xs)
    dh_next = [0.0] * cell.hidden_size
    for t in range(T - 1, -1, -1):
        proj.forward(h_outs[t])
        d_logits = loss.grad(logits_list[t], ys[t])
        dh_from_proj = proj.backward(d_logits)
        _, dh_next = cell.backward(dh_from_proj, dh_next)


def gradient_check_bptt(seed=7, T=3, V=8, H=5, eps=1e-5):
    random.seed(seed)
    cell = nn.RNNCell(V, H)
    proj = nn.Linear(H, V)
    loss = nn.SoftmaxCrossEntropy()
    xs = [one_hot(random.randrange(V), V) for _ in range(T)]
    ys = [random.randrange(V) for _ in range(T)]

    unrolled_grads(cell, proj, loss, xs, ys)
    analytic = []
    for layer in (cell, proj):
        for _, g in layer.parameters():
            analytic.extend(g)

    numeric = []
    for layer in (cell, proj):
        for v, _ in layer.parameters():
            for k in range(len(v)):
                orig = v[k]
                v[k] = orig + eps
                lp = unrolled_loss(cell, proj, loss, xs, ys)
                v[k] = orig - eps
                lm = unrolled_loss(cell, proj, loss, xs, ys)
                v[k] = orig
                numeric.append((lp - lm) / (2 * eps))

    worst = 0.0
    for a, n in zip(analytic, numeric):
        denom = max(1.0, abs(a), abs(n))
        worst = max(worst, abs(a - n) / denom)
    return worst


gc_worst = gradient_check_bptt()
print(f"unrolled-BPTT worst relative gradient error: {gc_worst:.2e}")""")

md(r"""## Vanishing gradients: the Jacobian-product norm

The price of the recurrent prior. Each backward step through time
multiplies the gradient on the hidden state by the Jacobian

$$\frac{\partial h_{t+1}}{\partial h_t}
   = \operatorname{diag}\!\bigl(1 - h_{t+1}^2\bigr)\, W_{hh}.$$

Propagating a gradient back `k` steps multiplies `k` such Jacobians. We
measure how the norm of that product behaves as the unroll depth grows: run
the cell forward on a random input stream to get realistic hidden states,
then push a unit gradient backward through the recurrence alone (no output
injections) and record its L2 norm after each step. With a freshly
initialized cell the norm decays geometrically, which is the vanishing
gradient: early timesteps receive an exponentially attenuated signal from
later losses.""")

code(r"""def jacobian_norm_decay(seed=0, depth=40, H=64, V=75):
    random.seed(seed)
    cell = nn.RNNCell(V, H)

    # Forward on a random stream to populate realistic hidden states.
    cell.reset_cache()
    h_state = None
    for _ in range(depth):
        x = one_hot(random.randrange(V), V)
        _, h_state = cell.forward(x, h_state)

    # Push a unit gradient backward through the recurrence only: dh_out = 0
    # at every step, dstate_next carries the propagating gradient. Its norm
    # is the norm of the accumulated Jacobian product.
    dh_out_zero = [0.0] * H
    dstate = [1.0 / math.sqrt(H)] * H        # unit-norm seed
    norms = [math.sqrt(sum(d * d for d in dstate))]
    for _ in range(depth):
        _, dstate = cell.backward(dh_out_zero, dstate)
        norms.append(math.sqrt(sum(d * d for d in dstate)))
    return norms


vg_norms = jacobian_norm_decay()
print("step   ||gradient through recurrence||")
for k in (0, 1, 2, 5, 10, 20, 30, 40):
    print(f"{k:>4}   {vg_norms[k]:.3e}")
ratio = vg_norms[40] / vg_norms[0]
print(f"\nafter 40 steps the gradient norm is {ratio:.2e} of its starting value")""")

md(r"""## Sampling

Greedy-with-temperature sampling. Warm the state on a seed string, then
roll forward one character at a time at temperature 0.8, feeding each
sampled character back in. The samples are quoted in the chapter to show the
trajectory from letter frequencies to recognizable words to English-shaped
fragments that still drift over longer spans (the drift is the
vanishing-gradient limit showing in the output).""")

code(r"""def sample(cell, proj, seed_text, length=200, temperature=0.8):
    cell.reset_cache()
    h_state = None
    h_out = None
    for ch in seed_text:
        x = one_hot(c2i[ch], vocab_size)
        h_out, h_state = cell.forward(x, h_state)
    out = [seed_text]
    for _ in range(length):
        logits = proj.forward(h_out)
        probs = nn.softmax([z / temperature for z in logits])
        r = random.random()
        cum = 0.0
        idx = len(probs) - 1
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                idx = i
                break
        out.append(i2c[idx])
        x = one_hot(idx, vocab_size)
        h_out, h_state = cell.forward(x, h_state)
    cell.reset_cache()         # drop sampling caches before next training pass
    return "".join(out)""")

md(r"""## Training

15 epochs. Each epoch shuffles the chunks and runs one BPTT step per chunk;
the per-epoch mean loss is the mean over chunks of the per-character loss.
At each epoch boundary we sample 200 characters seeded with `"Alice "`.
This is pure Python: a vanilla recurrent net trained by hand-derived BPTT,
roughly twenty minutes end to end.""")

code(r"""EPOCHS, LR, SEED = 15, 0.5, 0

cell, proj = build_rnn(SEED)
loss = nn.SoftmaxCrossEntropy()
random.seed(SEED)        # drives the per-epoch chunk shuffle

history = []
samples = {}
for epoch in range(EPOCHS):
    random.shuffle(chunks)
    total = 0.0
    for ch_ids in chunks:
        total += train_one_chunk(cell, proj, loss, ch_ids, LR)
    avg = total / len(chunks)
    history.append(avg)

    random.seed(1234 + epoch)
    s = sample(cell, proj, "Alice ", length=200, temperature=0.8)
    samples[epoch] = s
    print(f"epoch {epoch:2d}  mean per-char loss {avg:.4f}")
    print(f"  sample: {s!r}\n")
    random.seed(SEED + epoch + 1)   # restore a deterministic shuffle stream""")

md(r"""## Results

The mean per-character loss starts near the uniform baseline `log 75 ~ 4.32`
nats and descends over the 15 epochs. The final loss converts to perplexity
by `exp(loss)`. Chapter 4's fixed-context model reached 2.04 nats in 4
epochs on independent windows; the recurrent network lands at a comparable
per-character loss but takes more epochs and pays for the sequential BPTT,
in exchange for an unbounded-in-principle memory. The full give-up/gain
comparison is in the chapter.""")

code(r"""final = history[-1]
print(f"{'epoch':<8}{'mean loss (nats/char)':>24}")
print("-" * 32)
for e, l in enumerate(history):
    print(f"{e:<8}{l:>24.4f}")
print()
print(f"uniform baseline:  {math.log(vocab_size):.4f} nats/char")
print(f"final loss:        {final:.4f} nats/char")
print(f"perplexity:        {math.exp(final):.4f}")
print(f"parameters:        {n_params}")
print(f"gradient_check:    {gc_worst:.2e}")
print(f"vanishing-grad:    norm after 40 steps is "
      f"{vg_norms[40] / vg_norms[0]:.2e} of start")""")

md(r"""## Figure: the loss trajectory

Mean per-character loss per epoch, with the uniform-random baseline marked
and Chapter 4's 4-epoch fixed-context endpoint shown for reference. The
curve is the visual statement of the chapter's experiment and of the
comparison: the recurrent network reaches a comparable per-character loss to
the fixed-context model, but over more epochs. Saved to
`../book/figures/ch05-rnn-loss.pdf` and placed in the chapter at the
experiment section.""")

code(r"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_PDF = "../book/figures/ch05-rnn-loss.pdf"
baseline = math.log(vocab_size)
BENGIO_FINAL = 2.04          # Chapter 4 fixed-context LM, 4 epochs
epochs_axis = list(range(1, len(history) + 1))

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(epochs_axis, history, marker="o", color="#C44E52",
        label="recurrent net (RNNCell 75->64)")
ax.axhline(baseline, linestyle="--", color="#999999",
           label=f"uniform baseline (log 75 = {baseline:.2f})")
ax.axhline(BENGIO_FINAL, linestyle=":", color="#4C72B0",
           label=f"fixed-context LM, 4 epochs ({BENGIO_FINAL:.2f})")
ax.annotate(f"{history[-1]:.2f} nats/char\n(perplexity {math.exp(final):.2f})",
            xy=(epochs_axis[-1], history[-1]),
            xytext=(-10, 30), textcoords="offset points",
            ha="right", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#333333"))
ax.set_xlabel("epoch")
ax.set_ylabel("mean per-character loss (nats)")
ax.set_title("Char-level Alice: recurrent network, loss per epoch")
ax.set_xticks(epochs_axis)
ax.set_ylim(1.8, baseline + 0.3)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
os.makedirs(os.path.dirname(FIG_PDF), exist_ok=True)
fig.savefig(FIG_PDF)
print(f"saved {FIG_PDF}")""")


nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}
with open("ch05-rnn.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote ch05-rnn.ipynb")
