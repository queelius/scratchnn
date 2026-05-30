"""Scaling experiment with a properly-sized transformer.

In `pointer_scaling.py` we used a d_model=32 transformer with 3000
iterations, and the transformer underperformed the MLP at every M.
This file rerun the experiment with a bigger transformer (d_model=64,
6000 iters), to test whether the earlier transformer underperformance
was due to insufficient compute or genuine architectural failure.

We also extend the M range to {16, 24, 32, 48, 64} to find the regime
where even hidden=128 MLPs run out of capacity (128 / 2**A hidden units
per address combination becomes a tight budget at A>=6).

Run: python examples/pointer_scaling_bigxf.py
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


def run(M_values=(16, 24, 32, 48, 64), n_train=20000, n_iters=6000,
        seed=0):
    rows = []
    for M in M_values:
        A = max(3, (M - 1).bit_length())
        T = M + A
        n_addr = 2 ** A
        print(f"\n{'=' * 70}")
        print(f"M = {M}, A = {A}, addresses = {n_addr}, "
              f"sequence length = {T+1}")
        print(f"  MLP hidden=128 -> {128 // n_addr} units per address combination")
        print(f"  input space size = 2^{T} = {2**T:,}")
        print(f"{'=' * 70}")

        X_tr, Y_tr = make_variant1(n_train, M=M, A=A, seed=seed)
        X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1)

        # MLP
        mlp = MLPBaseline(context_len=T, embed_dim=4, hidden=128, seed=seed)
        n_mlp = count_params(mlp)
        t0 = time.time()
        train_mlp_on_examples(mlp, X_tr, Y_tr, n_iters=n_iters,
                              batch_size=64, lr=1e-3, log_every=2000,
                              seed=seed, silent=True)
        mlp_acc, mlp_loss = eval_mlp(mlp, X_te, Y_te)
        print(f"\n  MLP  (params {n_mlp:>7,}):  test acc {mlp_acc:.4f}  "
              f"loss {mlp_loss:.4f}  ({time.time()-t0:.0f}s)")

        # Bigger 2-layer transformer
        xf = BitTransformer(d_model=64, n_heads=1, d_ff=128, n_layers=2,
                            max_T=T, seed=seed)
        n_xf = count_params(xf)
        t0 = time.time()
        train_transformer_on_examples(xf, X_tr, Y_tr, n_iters=n_iters,
                                       batch_size=32, lr=1e-3,
                                       log_every=2000, seed=seed,
                                       silent=True)
        xf_acc, xf_loss = eval_transformer(xf, X_te, Y_te)
        print(f"  XF   (params {n_xf:>7,}):  test acc {xf_acc:.4f}  "
              f"loss {xf_loss:.4f}  ({time.time()-t0:.0f}s)")

        rows.append({
            'M': M, 'A': A, 'mlp_params': n_mlp, 'mlp_acc': mlp_acc,
            'xf_params': n_xf, 'xf_acc': xf_acc,
        })

    print(f"\n{'=' * 70}\nSCALING SUMMARY (bigger transformer, more iters)\n{'=' * 70}")
    print(f"{'M':>4} {'A':>3} {'h/addr':>7} {'MLP acc':>9} {'XF acc':>9}")
    print("-" * 40)
    for r in rows:
        h_per_addr = 128 // (2 ** r['A'])
        print(f"{r['M']:>4} {r['A']:>3} {h_per_addr:>7} "
              f"{r['mlp_acc']:>9.4f} {r['xf_acc']:>9.4f}")
    return rows


if __name__ == "__main__":
    run()
