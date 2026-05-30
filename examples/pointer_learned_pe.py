"""Verify the learned-positional-encoding fix for the pointer task.

Agent 4's architecture sweep found that replacing the sinusoidal PE with a
learned PE lifts M=16 test accuracy from ~75% to 100%. This script
verifies that fix at the FULL set of M values (sample-efficiency and
scaling experiments) so we can replace the post's empirical claims.

LearnedPETransformer is identical to BitTransformer except the
sinusoidal_positions() encoding is replaced by a learned table of
shape (max_T, d_model), initialized small and registered as a model
parameter.

Run: python examples/pointer_learned_pe.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_pointer_dgp import make_variant1
from pointer_transformer import BitTransformer, MLPBaseline, count_params
from pointer_experiments import (
    train_transformer_on_examples, eval_transformer,
    train_mlp_on_examples, eval_mlp,
)
from transformer import (
    Embedding, Linear, LayerNorm, FFN, CausalMultiHeadAttention,
    TransformerBlock,
)


class LearnedPETransformer:
    """Decoder-only transformer with a LEARNED positional encoding."""

    def __init__(self, d_model=64, n_heads=1, d_ff=128, n_layers=2,
                 max_T=64, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = 2
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_T = max_T
        # Learned positional encoding (replaces sinusoidal).
        self.pe = rng.standard_normal((max_T, d_model)).astype(np.float32) * 0.02
        self.dpe = np.zeros_like(self.pe)
        # The rest is the same as BitTransformer.
        self.embed = Embedding(self.vocab_size, d_model, rng)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, rng)
                       for _ in range(n_layers)]
        self.ln_final = LayerNorm(d_model)
        self.head = Linear(d_model, self.vocab_size, rng)
        self._T = None  # cached actual sequence length

    def forward(self, ids):
        T = ids.shape[0]
        assert T <= self.max_T
        self._T = T
        x = self.embed.forward(ids) + self.pe[:T]
        for blk in self.blocks:
            x = blk.forward(x)
        return self.head.forward(self.ln_final.forward(x))

    def backward(self, grad):
        g = self.head.backward(grad)
        g = self.ln_final.backward(g)
        for blk in reversed(self.blocks):
            g = blk.backward(g)
        # g is now d/dx of embed(ids) + pe[:T]. Accumulate into both.
        self.dpe[:self._T] += g
        self.embed.backward(g)

    def params(self):
        out = [(self.pe, self.dpe)]
        out += self.embed.params() + self.ln_final.params() + self.head.params()
        for blk in self.blocks:
            out += blk.params()
        return out

    def zero_grad(self):
        for _, g in self.params():
            g.fill(0.0)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def sample_efficiency_with_learned_pe(M=16, A=4, n_iters=6000, seed=0):
    print(f"\n{'=' * 70}")
    print(f"SAMPLE EFFICIENCY (M={M}, A={A}, learned PE)")
    print(f"{'=' * 70}")
    T = M + A
    X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1000)

    rows = []
    for n_train in (200, 500, 1000, 2000, 5000, 20000):
        print(f"\nn_train = {n_train}")
        X_tr, Y_tr = make_variant1(n_train, M=M, A=A, seed=seed)

        # MLP for comparison.
        mlp = MLPBaseline(context_len=T, embed_dim=4, hidden=128, seed=seed)
        t0 = time.time()
        train_mlp_on_examples(mlp, X_tr, Y_tr, n_iters=n_iters,
                              batch_size=min(64, max(8, n_train // 8)),
                              lr=1e-3, log_every=2000, seed=seed,
                              silent=True)
        mlp_acc, mlp_loss = eval_mlp(mlp, X_te, Y_te)
        print(f"  MLP    (hidden=128, {count_params(mlp):>5,} p):  "
              f"acc {mlp_acc:.4f}  ({time.time()-t0:.0f}s)")

        # Learned-PE transformer.
        xf = LearnedPETransformer(d_model=64, n_heads=1, d_ff=128,
                                   n_layers=2, max_T=T, seed=seed)
        t0 = time.time()
        train_transformer_on_examples(xf, X_tr, Y_tr, n_iters=n_iters,
                                       batch_size=min(32, max(8, n_train // 16)),
                                       lr=1e-3, log_every=2000, seed=seed,
                                       silent=True)
        xf_acc, xf_loss = eval_transformer(xf, X_te, Y_te)
        print(f"  XF-LPE (d_model=64, {count_params(xf):>5,} p):  "
              f"acc {xf_acc:.4f}  ({time.time()-t0:.0f}s)")

        rows.append({'n_train': n_train, 'mlp_acc': mlp_acc,
                     'xf_acc': xf_acc})

    print(f"\n{'-' * 50}\nSAMPLE-EFFICIENCY (learned PE) SUMMARY\n{'-' * 50}")
    print(f"{'n_train':>8} {'MLP acc':>10} {'XF-LPE acc':>12}")
    for r in rows:
        print(f"{r['n_train']:>8} {r['mlp_acc']:>10.4f} {r['xf_acc']:>12.4f}")
    return rows


def scaling_with_learned_pe(M_values=(16, 24, 32, 48, 64), n_train=20000,
                            n_iters=6000, seed=0):
    print(f"\n{'=' * 70}")
    print(f"SCALING with LEARNED PE (n_train={n_train})")
    print(f"{'=' * 70}")

    rows = []
    for M in M_values:
        A = max(3, (M - 1).bit_length())
        T = M + A
        print(f"\nM={M}, A={A}, T={T}")
        X_tr, Y_tr = make_variant1(n_train, M=M, A=A, seed=seed)
        X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1)

        # MLP
        mlp = MLPBaseline(context_len=T, embed_dim=4, hidden=128, seed=seed)
        t0 = time.time()
        train_mlp_on_examples(mlp, X_tr, Y_tr, n_iters=n_iters,
                              batch_size=64, lr=1e-3, log_every=2000,
                              seed=seed, silent=True)
        mlp_acc, mlp_loss = eval_mlp(mlp, X_te, Y_te)
        print(f"  MLP    (hidden=128, {count_params(mlp):>6,} p):  "
              f"acc {mlp_acc:.4f}  ({time.time()-t0:.0f}s)")

        # Learned-PE transformer
        xf = LearnedPETransformer(d_model=64, n_heads=1, d_ff=128,
                                   n_layers=2, max_T=T, seed=seed)
        t0 = time.time()
        train_transformer_on_examples(xf, X_tr, Y_tr, n_iters=n_iters,
                                       batch_size=32, lr=1e-3,
                                       log_every=2000, seed=seed,
                                       silent=True)
        xf_acc, xf_loss = eval_transformer(xf, X_te, Y_te)
        print(f"  XF-LPE (d_model=64, {count_params(xf):>6,} p):  "
              f"acc {xf_acc:.4f}  ({time.time()-t0:.0f}s)")

        rows.append({'M': M, 'A': A, 'mlp_acc': mlp_acc, 'xf_acc': xf_acc})

    print(f"\n{'-' * 50}\nSCALING (learned PE) SUMMARY\n{'-' * 50}")
    print(f"{'M':>3} {'A':>3} {'MLP acc':>10} {'XF-LPE acc':>12}")
    for r in rows:
        print(f"{r['M']:>3} {r['A']:>3} {r['mlp_acc']:>10.4f} {r['xf_acc']:>12.4f}")
    return rows


if __name__ == "__main__":
    sample_efficiency_with_learned_pe()
    scaling_with_learned_pe()
