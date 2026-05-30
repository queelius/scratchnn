"""Sample efficiency: at fixed M, how much data does each model need?

If the transformer is learning the lookup algorithm (rather than
memorizing input-output pairs), it should generalize from much less
training data than the MLP. The MLP allocates roughly 128 / 2**A
hidden units per address combination; it needs enough examples per
address to learn the per-address extractor. The transformer's
content-addressable attention is a single algorithm that works for any
address, so few examples should suffice.

Fix M=16, A=4 (1M-input-space task, well beyond the regime where pure
memorization is feasible). Vary n_train across {200, 500, 1000, 2000,
5000, 20000}. Train each model for 6000 Adam iterations. Compare
held-out accuracy.

Run: python examples/pointer_sample_efficiency.py
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


def run(n_trains=(200, 500, 1000, 2000, 5000, 20000), M=16, A=4,
        n_iters=6000, seed=0):
    T = M + A  # context length (target is +1, not in context)
    X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1000)
    print(f"M={M}, A={A}, sequence length {M+A+1}, input space 2^{M+A} = {2**(M+A)}")
    print(f"Held-out test set: 2000 examples")
    print(f"Training: Adam lr=1e-3, {n_iters} iters\n")

    rows = []
    for n_train in n_trains:
        print(f"{'-' * 70}\nn_train = {n_train} "
              f"({100 * n_train / 2**(M+A):.4f}% of input space)\n{'-' * 70}")
        X_tr, Y_tr = make_variant1(n_train, M=M, A=A, seed=seed)

        # MLP (hidden=128)
        mlp = MLPBaseline(context_len=T, embed_dim=4, hidden=128, seed=seed)
        n_mlp = count_params(mlp)
        t0 = time.time()
        train_mlp_on_examples(mlp, X_tr, Y_tr, n_iters=n_iters,
                              batch_size=min(64, max(8, n_train // 8)),
                              lr=1e-3, log_every=2000,
                              seed=seed, silent=True)
        mlp_acc, mlp_loss = eval_mlp(mlp, X_te, Y_te)
        t_mlp = time.time() - t0
        print(f"  MLP  (params {n_mlp:>6,}):  test acc {mlp_acc:.4f}  "
              f"loss {mlp_loss:.4f}  ({t_mlp:.0f}s)")

        # 2-layer transformer (d_model=64)
        xf = BitTransformer(d_model=64, n_heads=1, d_ff=128, n_layers=2,
                            max_T=T, seed=seed)
        n_xf = count_params(xf)
        t0 = time.time()
        train_transformer_on_examples(xf, X_tr, Y_tr, n_iters=n_iters,
                                       batch_size=min(32, max(8, n_train // 16)),
                                       lr=1e-3, log_every=2000,
                                       seed=seed, silent=True)
        xf_acc, xf_loss = eval_transformer(xf, X_te, Y_te)
        t_xf = time.time() - t0
        print(f"  XF   (params {n_xf:>6,}):  test acc {xf_acc:.4f}  "
              f"loss {xf_loss:.4f}  ({t_xf:.0f}s)")

        rows.append({
            'n_train': n_train, 'mlp_acc': mlp_acc, 'mlp_loss': mlp_loss,
            'xf_acc': xf_acc, 'xf_loss': xf_loss,
        })

    print(f"\n{'=' * 70}\nSAMPLE-EFFICIENCY SUMMARY (M=16, 6000 iters)\n{'=' * 70}")
    print(f"{'n_train':>8} {'%-space':>10} {'MLP acc':>9} {'XF acc':>9}")
    print("-" * 40)
    for r in rows:
        pct = 100 * r['n_train'] / 2 ** (M + A)
        print(f"{r['n_train']:>8} {pct:>9.4f}% {r['mlp_acc']:>9.4f} {r['xf_acc']:>9.4f}")
    return rows


if __name__ == "__main__":
    run()
