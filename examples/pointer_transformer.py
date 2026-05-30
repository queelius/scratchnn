"""Transformer experiments on the recursive pointer-dereferencing bit DGP.

Trains and evaluates four model variants on the bit stream from
`bit_dgp.py`:

  - MLP baseline (fixed-context window, no attention)
  - 1-layer 1-head transformer  (canonical "attention is lookup" demo)
  - 1-layer 2-head transformer  (head-specialization comparison)
  - 2-layer 1-head transformer  (depth experiment for multi-hop derefs)

All four are evaluated on the same held-out bit stream. The evaluation
breaks the per-position cross-entropy down by:
  (a) bit role (literal vs syntax-marker vs gamma-address vs deref-value)
  (b) hop depth for deref-value bits

The architecture pieces (Embedding, Linear, LayerNorm,
CausalMultiHeadAttention, FFN, TransformerBlock, Adam, etc.) are
imported from `transformer.py`. That file's pedagogical post is
separate from this one; we reuse its math here because the operators
are correct and gradient-checked, only the training task differs.

Run: python examples/pointer_transformer.py
"""
import math
import os
import sys
import time

import numpy as np

# Sibling imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bit_dgp import generate_stream
from transformer import (
    Embedding,
    Linear,
    LayerNorm,
    FFN,
    CausalMultiHeadAttention,
    TransformerBlock,
    sinusoidal_positions,
    softmax,
    softmax_cross_entropy,
    Adam,
)


# ============================================================
# Models
# ============================================================

class BitTransformer:
    """Decoder-only transformer over the 2-token bit vocabulary.

    Same architecture as the standard `Transformer` in `transformer.py`
    but pinned to a small vocabulary, with `n_heads` and `n_layers`
    exposed so we can run the head-specialization and depth experiments
    from one class.
    """

    def __init__(self, d_model=32, n_heads=1, d_ff=64, n_layers=1,
                 max_T=64, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = 2
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_T = max_T
        self.pe = sinusoidal_positions(max_T, d_model)
        self.embed = Embedding(self.vocab_size, d_model, rng)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, rng)
                       for _ in range(n_layers)]
        self.ln_final = LayerNorm(d_model)
        self.head = Linear(d_model, self.vocab_size, rng)

    def forward(self, ids):
        T = ids.shape[0]
        assert T <= self.max_T, f"sequence longer than max_T ({T} > {self.max_T})"
        x = self.embed.forward(ids) + self.pe[:T]
        for blk in self.blocks:
            x = blk.forward(x)
        return self.head.forward(self.ln_final.forward(x))

    def backward(self, grad):
        g = self.head.backward(grad)
        g = self.ln_final.backward(g)
        for blk in reversed(self.blocks):
            g = blk.backward(g)
        self.embed.backward(g)

    def params(self):
        out = self.embed.params() + self.ln_final.params() + self.head.params()
        for blk in self.blocks:
            out += blk.params()
        return out

    def zero_grad(self):
        for _, g in self.params():
            g.fill(0.0)

    def attention_at(self, ids, layer=0, head=0):
        """Run a forward pass and return the attention matrix for the
        requested (layer, head). Shape (T, T). Useful for visualization."""
        self.forward(ids)
        attn = self.blocks[layer].attn._attn  # (H, T, T) cached by forward
        return attn[head]


class MLPBaseline:
    """Predict the next bit from a fixed-context window of previous bits.

    Architecture (per-position): bit embeddings of length `embed_dim`,
    concatenated across the `context_len` past bits, then a Linear/ReLU/
    Linear stack ending in 2 logits.

    Trained and evaluated one position at a time. Parameter count is
    tunable through `embed_dim` and `hidden`, so we can match the
    transformer's count for fair comparisons.
    """

    def __init__(self, context_len=64, embed_dim=8, hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        self.context_len = context_len
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.embed = Embedding(2, embed_dim, rng)
        self.fc1 = Linear(context_len * embed_dim, hidden, rng)
        self.fc2 = Linear(hidden, 2, rng)
        self._pre_relu = None
        self._ctx_shape = None

    def forward_one(self, context_ids):
        """Predict the logits over the next bit, given a fixed-length context."""
        embedded = self.embed.forward(context_ids)        # (ctx, embed_dim)
        self._ctx_shape = embedded.shape
        flat = embedded.reshape(-1)
        h = self.fc1.forward(flat)
        self._pre_relu = h
        h = np.maximum(h, 0)
        return self.fc2.forward(h)

    def backward_one(self, grad_logits):
        d_h = self.fc2.backward(grad_logits)
        d_h = d_h * (self._pre_relu > 0)
        d_flat = self.fc1.backward(d_h)
        d_embedded = d_flat.reshape(self._ctx_shape)
        self.embed.backward(d_embedded)

    def params(self):
        return self.embed.params() + self.fc1.params() + self.fc2.params()

    def zero_grad(self):
        for _, g in self.params():
            g.fill(0.0)


def count_params(model):
    return sum(p.size for p, _ in model.params())


# ============================================================
# Data utilities
# ============================================================

def make_train_test_streams(num_train_instructions=8000,
                            num_test_instructions=2000,
                            p_deref=0.20, offset_mean=4.0, seed=0):
    train_bits, train_instrs, train_tags, train_hops = generate_stream(
        num_train_instructions, p_deref=p_deref, offset_mean=offset_mean,
        seed=seed)
    test_bits, test_instrs, test_tags, test_hops = generate_stream(
        num_test_instructions, p_deref=p_deref, offset_mean=offset_mean,
        seed=seed + 1)
    return (np.array(train_bits, dtype=np.int64),
            np.array(test_bits, dtype=np.int64),
            test_tags, test_hops)


def sample_transformer_batch(bits, T, batch_size, rng):
    """Sample `batch_size` random length-T windows. Inputs at positions
    [s, s+T); targets are next-bit at positions [s+1, s+T+1)."""
    starts = rng.integers(0, len(bits) - T - 1, size=batch_size)
    x = np.stack([bits[s:s + T] for s in starts])
    y = np.stack([bits[s + 1:s + T + 1] for s in starts])
    return x, y


# ============================================================
# Training
# ============================================================

def train_transformer(model, bits, T, n_iters=2000, batch_size=16, lr=3e-4,
                      log_every=200, seed=0):
    """Standard Adam training over a bit stream. Tracks running loss."""
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.params(), lr=lr)
    running = 0.0
    history = []
    start = time.time()
    for it in range(1, n_iters + 1):
        x_b, y_b = sample_transformer_batch(bits, T, batch_size, rng)
        model.zero_grad()
        total = 0.0
        for b in range(batch_size):
            logits = model.forward(x_b[b])
            loss, dlogits = softmax_cross_entropy(logits, y_b[b])
            total += loss
            model.backward(dlogits / batch_size)
        optimizer.step()
        running += total / batch_size
        if it % log_every == 0:
            avg = running / log_every
            history.append(avg)
            elapsed = time.time() - start
            print(f"    iter {it:4d}  loss {avg:.4f}  ({elapsed:.0f}s)")
            running = 0.0
    return history


def train_mlp(model, bits, n_iters=2000, batch_size=64, lr=1e-3,
              log_every=200, seed=0):
    """Train the MLP baseline. Each iteration samples `batch_size` random
    positions and averages the per-example loss."""
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.params(), lr=lr)
    K = model.context_len
    n = len(bits)
    running = 0.0
    history = []
    start = time.time()
    for it in range(1, n_iters + 1):
        # Sample valid target positions: need K bits of past context.
        targets = rng.integers(K, n, size=batch_size)
        model.zero_grad()
        total = 0.0
        for t in targets:
            ctx = bits[t - K:t]
            y = int(bits[t])
            logits = model.forward_one(ctx)
            # Stable cross-entropy with grad.
            logits_shift = logits - logits.max()
            probs = np.exp(logits_shift)
            probs /= probs.sum()
            loss = -math.log(max(probs[y], 1e-12))
            grad = probs.copy()
            grad[y] -= 1.0
            grad /= batch_size  # accumulate mean
            model.backward_one(grad)
            total += loss
        optimizer.step()
        running += total / batch_size
        if it % log_every == 0:
            avg = running / log_every
            history.append(avg)
            elapsed = time.time() - start
            print(f"    iter {it:4d}  loss {avg:.4f}  ({elapsed:.0f}s)")
            running = 0.0
    return history


# ============================================================
# Evaluation
# ============================================================

def evaluate_transformer(model, bits, tags, hops, T, n_samples=400, seed=0):
    """Sample positions in the held-out stream; for each, compute the loss
    of the model's prediction at that position. Aggregate by bit tag and
    by deref hop depth."""
    rng = np.random.default_rng(seed)
    n = len(bits)
    valid = np.arange(T, n)  # need T bits of history before the target
    targets = rng.choice(valid, size=min(n_samples, len(valid)),
                         replace=False)
    by_tag = {}
    by_hop = {}
    losses = []
    for t in targets:
        ctx = bits[t - T:t]
        y = int(bits[t])
        logits = model.forward(ctx)
        last = logits[-1]
        m = float(last.max())
        e = np.exp(last - m)
        z = float(e.sum())
        loss = float(m - last[y] + math.log(z))
        losses.append(loss)
        by_tag.setdefault(tags[t], []).append(loss)
        if hops[t] is not None:
            by_hop.setdefault(hops[t], []).append(loss)
    return losses, by_tag, by_hop


def evaluate_mlp(model, bits, tags, hops, n_samples=400, seed=0):
    rng = np.random.default_rng(seed)
    K = model.context_len
    n = len(bits)
    valid = np.arange(K, n)
    targets = rng.choice(valid, size=min(n_samples, len(valid)),
                         replace=False)
    by_tag = {}
    by_hop = {}
    losses = []
    for t in targets:
        ctx = bits[t - K:t]
        y = int(bits[t])
        logits = model.forward_one(ctx)
        m = float(logits.max())
        e = np.exp(logits - m)
        z = float(e.sum())
        loss = float(m - logits[y] + math.log(z))
        losses.append(loss)
        by_tag.setdefault(tags[t], []).append(loss)
        if hops[t] is not None:
            by_hop.setdefault(hops[t], []).append(loss)
    return losses, by_tag, by_hop


def fmt_breakdown(losses, by_tag, by_hop):
    out = []
    out.append(f"  overall mean loss: {np.mean(losses):.4f}")
    out.append("  by bit role:")
    for tag in sorted(by_tag.keys()):
        ls = by_tag[tag]
        out.append(f"    {tag:<18} n={len(ls):>4}  mean={np.mean(ls):.4f}")
    out.append("  by hop depth (deref-value bits only):")
    for hop in sorted(by_hop.keys()):
        ls = by_hop[hop]
        out.append(f"    depth={hop}  n={len(ls):>4}  mean={np.mean(ls):.4f}")
    return "\n".join(out)


# ============================================================
# Main experiment
# ============================================================

def run_all(num_train=20000, num_test=4000, T=64, n_iters=2000, seed=0):
    print("Generating bit-DGP streams...")
    train_bits, test_bits, test_tags, test_hops = make_train_test_streams(
        num_train_instructions=num_train,
        num_test_instructions=num_test,
        p_deref=0.30, offset_mean=4.0, seed=seed)
    print(f"  train: {len(train_bits)} bits, test: {len(test_bits)} bits")

    # Distribution of hop depths in the test stream.
    n_deref_value_bits = sum(1 for h in test_hops if h is not None)
    hop_hist = {}
    for h in test_hops:
        if h is not None:
            hop_hist[h] = hop_hist.get(h, 0) + 1
    print(f"  test deref-value bits: {n_deref_value_bits}")
    print(f"  hop-depth distribution: {dict(sorted(hop_hist.items()))}")

    results = {}

    print("\n[1/4] MLP baseline...")
    # embed=4, hidden=32 gives ~8.3k params, matching the transformer's ~8.7k.
    mlp = MLPBaseline(context_len=T, embed_dim=4, hidden=32, seed=seed)
    print(f"  parameters: {count_params(mlp)}")
    train_mlp(mlp, train_bits, n_iters=n_iters, lr=1e-3, seed=seed)
    losses, by_tag, by_hop = evaluate_mlp(mlp, test_bits, test_tags,
                                          test_hops, n_samples=400,
                                          seed=seed + 10)
    print(fmt_breakdown(losses, by_tag, by_hop))
    results['mlp'] = {'overall': np.mean(losses), 'by_tag': by_tag,
                      'by_hop': by_hop}

    print("\n[2/4] 1-layer 1-head transformer...")
    t11 = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=1,
                         max_T=T, seed=seed)
    print(f"  parameters: {count_params(t11)}")
    train_transformer(t11, train_bits, T=T, n_iters=n_iters, lr=3e-3,
                      seed=seed)
    losses, by_tag, by_hop = evaluate_transformer(t11, test_bits, test_tags,
                                                  test_hops, T=T,
                                                  n_samples=400,
                                                  seed=seed + 10)
    print(fmt_breakdown(losses, by_tag, by_hop))
    results['t11'] = {'overall': np.mean(losses), 'by_tag': by_tag,
                      'by_hop': by_hop, 'model': t11}

    print("\n[3/4] 1-layer 2-head transformer...")
    t12 = BitTransformer(d_model=32, n_heads=2, d_ff=64, n_layers=1,
                         max_T=T, seed=seed)
    print(f"  parameters: {count_params(t12)}")
    train_transformer(t12, train_bits, T=T, n_iters=n_iters, lr=3e-3,
                      seed=seed)
    losses, by_tag, by_hop = evaluate_transformer(t12, test_bits, test_tags,
                                                  test_hops, T=T,
                                                  n_samples=400,
                                                  seed=seed + 10)
    print(fmt_breakdown(losses, by_tag, by_hop))
    results['t12'] = {'overall': np.mean(losses), 'by_tag': by_tag,
                      'by_hop': by_hop, 'model': t12}

    print("\n[4/4] 2-layer 1-head transformer...")
    t21 = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=2,
                         max_T=T, seed=seed)
    print(f"  parameters: {count_params(t21)}")
    train_transformer(t21, train_bits, T=T, n_iters=n_iters, lr=3e-3,
                      seed=seed)
    losses, by_tag, by_hop = evaluate_transformer(t21, test_bits, test_tags,
                                                  test_hops, T=T,
                                                  n_samples=400,
                                                  seed=seed + 10)
    print(fmt_breakdown(losses, by_tag, by_hop))
    results['t21'] = {'overall': np.mean(losses), 'by_tag': by_tag,
                      'by_hop': by_hop, 'model': t21}

    # Side-by-side summary table.
    print("\n" + "=" * 70)
    print("SUMMARY (mean cross-entropy in nats; lower is better)")
    print("=" * 70)
    header = ("position type".ljust(22)
              + "MLP".rjust(10) + "T(1L,1H)".rjust(12)
              + "T(1L,2H)".rjust(12) + "T(2L,1H)".rjust(12))
    print(header)
    print("-" * len(header))
    all_tags = set()
    for name in ['mlp', 't11', 't12', 't21']:
        all_tags.update(results[name]['by_tag'].keys())
    for tag in sorted(all_tags):
        row = tag.ljust(22)
        for name in ['mlp', 't11', 't12', 't21']:
            vals = results[name]['by_tag'].get(tag, [])
            row += (f"{np.mean(vals):.4f}".rjust(12) if vals
                    else "-".rjust(12))
        print(row)
    # Hop-depth breakdown.
    print()
    all_hops = set()
    for name in ['mlp', 't11', 't12', 't21']:
        all_hops.update(results[name]['by_hop'].keys())
    print("Deref-value loss by hop depth:")
    for hop in sorted(all_hops):
        row = f"  hop={hop}".ljust(22)
        for name in ['mlp', 't11', 't12', 't21']:
            vals = results[name]['by_hop'].get(hop, [])
            row += (f"{np.mean(vals):.4f}".rjust(12) if vals
                    else "-".rjust(12))
        print(row)
    print()
    print("Overall:")
    row = "".ljust(22)
    for name in ['mlp', 't11', 't12', 't21']:
        row += f"{results[name]['overall']:.4f}".rjust(12)
    print(row)

    return results


if __name__ == "__main__":
    run_all()
