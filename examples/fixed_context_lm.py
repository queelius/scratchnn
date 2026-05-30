"""Bengio 2003-style fixed-context neural language model on Alice.

Architecture, expressed as a vanilla `Network`:

    EmbedConcat(vocab, embed_dim, context_len)
    Linear(context_len * embed_dim, hidden_size)
    Tanh()
    Linear(hidden_size, vocab)
    SoftmaxCrossEntropy()

The `EmbedConcat` layer wraps an `Embedding` lookup plus the context
concatenation; downstream layers see a flat `list[float]` and the rest
of the library composes without change. Training uses `Network.fit`
directly: each example is a (context_ids, target_id) pair where
`context_ids` is a `list[int]` of length `context_len` and `target_id`
is an `int`.

Same Alice corpus as the RNN post, for a direct comparison: the RNN
carries unbounded past through a hidden state; this model carries no
state at all, only the last N tokens, but knows their *positional*
identities within the window.

Run:
    python examples/fixed_context_lm.py
"""
import os
import random
import sys

# Allow `import` of sibling files when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scratchnn as nn
from text_rnn import load_alice, build_vocab


def predicted_id(probs):
    return max(range(len(probs)), key=lambda i: probs[i])


def sample(net, seed_text, c2i, i2c, length, context_size,
           temperature=0.8):
    """Generate `length` characters starting from `seed_text`."""
    pad_char = " "
    if len(seed_text) < context_size:
        seed_text = pad_char * (context_size - len(seed_text)) + seed_text
    output = list(seed_text)
    for _ in range(length):
        context = output[-context_size:]
        ctx_ids = [c2i.get(c, c2i[pad_char]) for c in context]
        logits = net.forward(ctx_ids)
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
    return "".join(output)


def train(text, context_size=8, embed_dim=16, hidden_size=64, epochs=4,
          lr=0.02, batch_size=1, sample_every=1,
          sample_seed_text="alice was ", sample_length=200, seed=0):
    chars, c2i, i2c = build_vocab(text)
    vocab_size = len(chars)
    ids = [c2i[c] for c in text]

    print(f"corpus: {len(text)} chars; vocab: {vocab_size}")
    print(f"context N={context_size}, embed={embed_dim}, hidden={hidden_size}")

    random.seed(seed)
    net = nn.Network([
        nn.EmbedConcat(vocab_size, embed_dim, context_size),
        nn.Linear(context_size * embed_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, vocab_size),
    ], nn.SoftmaxCrossEntropy())

    n_params = sum(len(v) for v, _ in net.parameters())
    print(f"parameters: {n_params}")

    X, Y = [], []
    for i in range(len(ids) - context_size):
        X.append(ids[i:i + context_size])
        Y.append(ids[i + context_size])
    print(f"examples: {len(X)}\n")

    def callback(epoch, network, history):
        if not sample_every or (epoch + 1) % sample_every != 0:
            return
        print(f"  epoch {epoch:3d}  mean loss {history[-1]:.4f}")
        s = sample(network, sample_seed_text, c2i, i2c,
                   length=sample_length, context_size=context_size,
                   temperature=0.8)
        print(f"  sample: {s!r}\n")

    history = net.fit(X, Y, epochs=epochs, lr=lr, batch_size=batch_size,
                      callback=callback)
    return net, c2i, i2c, history


def main():
    text = load_alice(max_chars=30000)
    train(text, context_size=8, embed_dim=16, hidden_size=64, epochs=4,
          lr=0.02, batch_size=1, sample_every=1,
          sample_seed_text="alice was ", sample_length=200)


if __name__ == "__main__":
    main()
