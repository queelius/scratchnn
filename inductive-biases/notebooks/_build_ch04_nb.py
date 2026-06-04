"""Construct notebooks/ch04-fixed-context-lm.ipynb via nbformat.

Run this, then execute the notebook with nbconvert so every cell carries a
stored output. Kept in the repo so the notebook is reproducible from a
structured source rather than hand-edited JSON.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(text):
    cells.append(new_code_cell(text))


md(r"""# Chapter 4: Fixed-Context Language Models

Paired notebook for Chapter 4 of *Inductive Biases in Neural Networks*.
Trains Bengio's 2003 fixed-context neural language model on a char-level
subset of *Alice's Adventures in Wonderland*: embed the last `N` tokens
through a shared lookup table, concatenate, and feed a position-sensitive
MLP head that emits logits over the vocabulary. `SoftmaxCrossEntropy` is
the output head. Pure-Python `scratchnn` throughout, model and experiment.

**Seeds:** `random.seed(0)` before the network is built (this drives both
weight initialization and the per-example SGD order); each per-epoch
sample re-seeds `random.seed(1234 + epoch)` so the quoted samples are
reproducible. The gradient check uses `random.seed(7)`.

**Figure saved:** `../book/figures/ch04-lm-loss.pdf`

**Headline numbers (regenerated below):** 30,000 chars, 75-char vocab,
29,992 windows; `N = 8`, `d = 16`, `h = 64`; 14,331 parameters. Mean
per-character loss falls from the uniform baseline `log 75 ~ 4.32` nats to
**2.04** nats/char (perplexity **7.67**) in 4 epochs. That is **2.94 bits
per character** under a Shannon-optimal code, the chapter's
compression-as-prediction north star. `gradient_check` on the
`EmbedConcat`-headed network is on the order of 1e-9.""")

md(r"""## Setup

The library is pure Python (`math`, `random` only). The corpus loader and
vocabulary builder come from `examples/text_rnn.py`, which reads
`examples/data/alice.txt` (Project Gutenberg eBook #11, header and footer
stripped) and takes the first 30,000 characters. The same corpus and
vocabulary feed the recurrent-network chapter, so the two models are
trained on identical data.""")

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

The whole model is a plain `Network`. `EmbedConcat(vocab, d, N)` looks up
the `N` context ids in a shared `Embedding` table and concatenates them
into a flat vector of length `N * d`; from there it is an ordinary MLP
head (`Linear -> Tanh -> Linear`) emitting logits. No custom training
loop and no recurrence.

The parameter count breaks down as the embedding table, the first
`Linear`, its bias, the second `Linear`, and its bias:

$$75 \cdot 16 \;+\; 8 \cdot 16 \cdot 64 \;+\; 64 \;+\; 64 \cdot 75 \;+\; 75 \;=\; 14{,}331.$$""")

code(r"""N, EMBED, HIDDEN = 8, 16, 64


def build_lm(seed):
    random.seed(seed)
    return nn.Network([
        nn.EmbedConcat(vocab_size, EMBED, N),
        nn.Linear(N * EMBED, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, vocab_size),
    ], nn.SoftmaxCrossEntropy())


n_params = sum(len(v) for v, _ in build_lm(0).parameters())
hand_count = (vocab_size * EMBED + N * EMBED * HIDDEN + HIDDEN
              + HIDDEN * vocab_size + vocab_size)
print(f"parameters (from net):   {n_params}")
print(f"parameters (hand count): {hand_count}")""")

md(r"""## Training data

Slide a window of length `N` over the corpus. Each example is the `N`
context ids in and the next id out. Unlike the recurrent setup, every
window-target pair is independent: there is no inter-example state to
carry, so this is ordinary supervised classification over `vocab_size`
classes.""")

code(r"""X, Y = [], []
for i in range(len(ids) - N):
    X.append(ids[i:i + N])
    Y.append(ids[i + N])
print(f"examples: {len(X)} (window {N} -> next char)")""")

md(r"""## Gradient check on `EmbedConcat`

The `EmbedConcat` forward concatenates `N` embedding lookups; its backward
slices the upstream gradient into `N` chunks and routes each back through
the embedding in LIFO order. A small `EmbedConcat`-headed network is
checked against a central-difference numerical gradient (the same
`gradient_check` used in Chapter 1). The worst relative error should sit
near machine precision, far under the `1e-4` tolerance.""")

code(r"""random.seed(7)
gc_net = nn.Network([
    nn.EmbedConcat(12, 4, 3),
    nn.Linear(3 * 4, 8),
    nn.Tanh(),
    nn.Linear(8, 12),
], nn.SoftmaxCrossEntropy())
gc_x = [random.randrange(12) for _ in range(3)]
gc_worst = nn.gradient_check(gc_net, gc_x, 5)
print(f"EmbedConcat worst relative gradient error: {gc_worst:.2e}")""")

md(r"""## Sampling

Greedy-with-temperature sampling. Seed with `"alice was "`, left-pad to the
context length if needed, and roll forward one character at a time at
temperature 0.8. The samples are quoted in the chapter to show the
trajectory from letter frequencies to recognizable words to
English-shaped fragments.""")

code(r"""def sample(net, seed_text, length=200, temperature=0.8):
    pad = " "
    if len(seed_text) < N:
        seed_text = pad * (N - len(seed_text)) + seed_text
    out = list(seed_text)
    for _ in range(length):
        ctx_ids = [c2i.get(c, c2i[pad]) for c in out[-N:]]
        logits = net.forward(ctx_ids)
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
    return "".join(out)""")

md(r"""## Training

SGD with learning rate 0.02, batch size 1, 4 epochs over all 29,992
windows. The `Network.fit` callback records the per-epoch mean loss and
draws one sample. This is the same `fit` loop every earlier model used;
nothing here knows it is a language model. A few minutes in pure Python.""")

code(r"""EPOCHS, LR, BATCH, SEED = 4, 0.02, 1, 0

net = build_lm(SEED)
samples = {}


def callback(epoch, network, history):
    random.seed(1234 + epoch)
    s = sample(network, "alice was ")
    samples[epoch] = s
    print(f"epoch {epoch}  mean loss {history[-1]:.4f}")
    print(f"  sample: {s!r}\n")


history = net.fit(X, Y, epochs=EPOCHS, lr=LR, batch_size=BATCH,
                  callback=callback)""")

md(r"""## Results

The mean per-character loss starts near the uniform baseline `log 75 ~
4.32` nats and descends across the four epochs. The final loss converts to
perplexity by `exp(loss)` and to bits per character by `loss / ln 2`. The
bits-per-character figure is not an analogy: cross-entropy in bits per
character *is* the code length per character under a Shannon-optimal code
built from the model's predictions. A model that predicts better
compresses better.""")

code(r"""final = history[-1]
print(f"{'epoch':<8}{'mean loss (nats/char)':>24}")
print("-" * 32)
for e, l in enumerate(history):
    print(f"{e:<8}{l:>24.4f}")
print()
print(f"uniform baseline:  {math.log(vocab_size):.4f} nats/char")
print(f"final loss:        {final:.4f} nats/char")
print(f"perplexity:        {math.exp(final):.4f}")
print(f"bits per char:     {final / math.log(2):.4f}")
print(f"parameters:        {n_params}")
print(f"gradient_check:    {gc_worst:.2e}")""")

md(r"""## Figure: the loss trajectory

Mean per-character loss per epoch, with the uniform-random baseline marked.
The curve is the visual statement of the chapter's experiment: a model that
sees only the last eight characters drops from chance to roughly 2 nats in
four passes. Saved to `../book/figures/ch04-lm-loss.pdf` and placed in the
chapter at the experiment section.""")

code(r"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_PDF = "../book/figures/ch04-lm-loss.pdf"
baseline = math.log(vocab_size)
epochs_axis = list(range(1, len(history) + 1))

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(epochs_axis, history, marker="o", color="#4C72B0",
        label="fixed-context LM (N=8)")
ax.axhline(baseline, linestyle="--", color="#999999",
           label=f"uniform baseline (log 75 = {baseline:.2f})")
ax.annotate(f"{history[-1]:.2f} nats/char\n(perplexity {math.exp(final):.2f})",
            xy=(epochs_axis[-1], history[-1]),
            xytext=(-10, 28), textcoords="offset points",
            ha="right", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#333333"))
ax.set_xlabel("epoch")
ax.set_ylabel("mean per-character loss (nats)")
ax.set_title("Char-level Alice: loss per epoch")
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
with open("ch04-fixed-context-lm.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote ch04-fixed-context-lm.ipynb")
