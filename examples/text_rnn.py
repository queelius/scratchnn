"""Char-level RNN on Lewis Carroll's *Alice's Adventures in Wonderland*.

The demo for the RNN post. Trains a vanilla `RNNCell` plus a `Linear`
output projection to predict the next character given the previous one,
unrolled over short windows via backprop-through-time. Samples generated
text periodically so the reader can watch the model go from random
characters to recognizable English words to Carroll-style prose
fragments.

Data: `examples/data/alice.txt`, downloaded from Project Gutenberg
(eBook #11). The Project Gutenberg header and footer are stripped at
load time.

Run:
    python examples/text_rnn.py
"""
import math
import os
import random
import sys

import scratchnn as nn

# Allow `import` of sibling files when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALICE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "alice.txt")


# ----- data ---------------------------------------------------------------

def load_alice(path=ALICE_PATH, max_chars=None):
    """Read Alice and strip Project Gutenberg header/footer."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    s = text.find(start_marker)
    e = text.find(end_marker)
    if s != -1 and e != -1:
        s = text.find("\n", s) + 1
        text = text[s:e]
    text = text.strip()
    if max_chars is not None:
        text = text[:max_chars]
    return text


def build_vocab(text):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    return chars, c2i, i2c


def one_hot(idx, n):
    v = [0.0] * n
    v[idx] = 1.0
    return v


# ----- training (BPTT, batch size 1) --------------------------------------

def zero_grads(cell, proj):
    for _, g in cell.parameters():
        for k in range(len(g)):
            g[k] = 0.0
    for _, g in proj.parameters():
        for k in range(len(g)):
            g[k] = 0.0


def sgd_step(cell, proj, lr, n):
    """Apply the *mean* accumulated gradient: same convention as
    `Network.step`. Gradients in `cell` accumulate across all T timesteps
    of BPTT (because the recurrent weights are reused at every step), so
    dividing by T turns the sum back into a mean and keeps the effective
    learning rate sane regardless of unroll length."""
    factor = lr / n
    for v, g in cell.parameters():
        for k in range(len(v)):
            v[k] -= factor * g[k]
    for v, g in proj.parameters():
        for k in range(len(v)):
            v[k] -= factor * g[k]


def clip_grads(cell, proj, max_norm=5.0):
    """Clip global gradient L2 norm to `max_norm`. Standard mitigation for
    exploding gradients in vanilla RNNs."""
    total = 0.0
    for v, g in cell.parameters():
        for gk in g:
            total += gk * gk
    for v, g in proj.parameters():
        for gk in g:
            total += gk * gk
    norm = math.sqrt(total)
    if norm > max_norm:
        scale = max_norm / norm
        for v, g in cell.parameters():
            for k in range(len(g)):
                g[k] *= scale
        for v, g in proj.parameters():
            for k in range(len(g)):
                g[k] *= scale


def train_one_chunk(cell, proj, loss, chunk_ids, vocab_size, lr):
    """One unrolled BPTT step on a single chunk of `len(chunk_ids)` ids.

    Each timestep predicts the next id from the current. Returns the
    mean per-character loss over the chunk.
    """
    T = len(chunk_ids) - 1

    zero_grads(cell, proj)
    cell.reset_cache()

    h_outs = []
    logits_list = []
    targets = []
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

    H = cell.hidden_size
    dh_next = [0.0] * H
    for t in range(T - 1, -1, -1):
        # Re-run proj.forward to refresh its cached input for this step.
        proj.forward(h_outs[t])
        d_logits = loss.grad(logits_list[t], targets[t])
        dh_from_proj = proj.backward(d_logits)
        _, dh_next = cell.backward(dh_from_proj, dh_next)

    clip_grads(cell, proj, max_norm=5.0)
    sgd_step(cell, proj, lr, T)
    return L / T


# ----- sampling -----------------------------------------------------------

def sample(cell, proj, seed_text, c2i, i2c, length, temperature=0.8):
    """Generate `length` characters starting from `seed_text`."""
    vocab_size = len(c2i)
    h_state = None
    h_out = None
    for ch in seed_text:
        x = one_hot(c2i[ch], vocab_size)
        h_out, h_state = cell.forward(x, h_state)
    output = [seed_text]
    for _ in range(length):
        logits = proj.forward(h_out)
        # Temperature softmax: smaller temperature is more conservative.
        scaled = [z / temperature for z in logits]
        probs = nn.softmax(scaled)
        r = random.random()
        cum = 0.0
        idx = len(probs) - 1
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                idx = i
                break
        output.append(i2c[idx])
        x = one_hot(idx, vocab_size)
        h_out, h_state = cell.forward(x, h_state)
    # Drop the caches the sampling forwards just pushed; they would otherwise
    # poison the next training pass.
    cell.reset_cache()
    return "".join(output)


# ----- top level ----------------------------------------------------------

def train(text, hidden_size=64, seq_len=32, epochs=15, lr=0.5,
          sample_every=1, sample_seed_text="Alice ", sample_length=200,
          seed=0):
    chars, c2i, i2c = build_vocab(text)
    vocab_size = len(chars)
    print(f"corpus: {len(text)} chars; vocab: {vocab_size} chars")

    random.seed(seed)
    cell = nn.RNNCell(vocab_size, hidden_size)
    proj = nn.Linear(hidden_size, vocab_size)
    loss = nn.SoftmaxCrossEntropy()

    # Chunk the corpus into non-overlapping windows of length seq_len + 1
    # (so each window has seq_len input/target pairs).
    ids = [c2i[c] for c in text]
    chunks = []
    for i in range(0, len(ids) - seq_len, seq_len):
        chunks.append(ids[i:i + seq_len + 1])
    print(f"chunks: {len(chunks)} of length {seq_len + 1}")

    history = []
    for epoch in range(epochs):
        random.shuffle(chunks)
        total = 0.0
        for ch_ids in chunks:
            total += train_one_chunk(cell, proj, loss, ch_ids, vocab_size, lr)
        avg = total / len(chunks)
        history.append(avg)
        print(f"  epoch {epoch:3d}  mean per-char loss {avg:.4f}")
        if sample_every and (epoch + 1) % sample_every == 0:
            s = sample(cell, proj, sample_seed_text, c2i, i2c,
                       length=sample_length, temperature=0.8)
            print(f"  sample: {s!r}")
            print()

    return cell, proj, c2i, i2c, history


def main():
    text = load_alice(max_chars=30000)  # subset; full text ~140KB
    train(text, hidden_size=64, seq_len=32, epochs=15, lr=0.5,
          sample_every=1, sample_seed_text="Alice ", sample_length=200)


if __name__ == "__main__":
    main()
