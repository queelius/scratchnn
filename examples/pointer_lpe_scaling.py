"""Focused scaling verification: does learned PE solve the lookup at higher M?

Agent 4's architecture sweep showed learned PE -> 100% at M=16. This is
the minimal follow-up to confirm the fix generalizes to M={16, 24, 32}.
We skip the full sample-efficiency curve and other M values to keep
compute tractable.

Run: python examples/pointer_lpe_scaling.py
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
from pointer_learned_pe import LearnedPETransformer


def main(M_values=(16, 24, 32), n_train=20000, n_iters=6000, seed=0):
    print(f"Focused learned-PE scaling: M in {M_values}, "
          f"n_train={n_train}, iters={n_iters}\n")

    rows = []
    for M in M_values:
        A = max(3, (M - 1).bit_length())
        T = M + A
        print(f"{'-' * 50}\nM={M}, A={A}, T={T}\n{'-' * 50}")
        X_tr, Y_tr = make_variant1(n_train, M=M, A=A, seed=seed)
        X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1)

        # Baseline sinusoidal-PE transformer for reference.
        xf_sin = BitTransformer(d_model=64, n_heads=1, d_ff=128, n_layers=2,
                                max_T=T, seed=seed)
        t0 = time.time()
        train_transformer_on_examples(xf_sin, X_tr, Y_tr, n_iters=n_iters,
                                       batch_size=32, lr=1e-3,
                                       log_every=2000, seed=seed,
                                       silent=True)
        sin_acc, _ = eval_transformer(xf_sin, X_te, Y_te)
        print(f"  XF (sinusoidal PE): acc {sin_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")

        # Learned-PE transformer.
        xf_lpe = LearnedPETransformer(d_model=64, n_heads=1, d_ff=128,
                                       n_layers=2, max_T=T, seed=seed)
        t0 = time.time()
        train_transformer_on_examples(xf_lpe, X_tr, Y_tr, n_iters=n_iters,
                                       batch_size=32, lr=1e-3,
                                       log_every=2000, seed=seed,
                                       silent=True)
        lpe_acc, _ = eval_transformer(xf_lpe, X_te, Y_te)
        print(f"  XF (LEARNED PE):    acc {lpe_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")

        rows.append({'M': M, 'sinusoidal': sin_acc, 'learned': lpe_acc})

    print(f"\n{'=' * 50}\nSUMMARY\n{'=' * 50}")
    print(f"{'M':>3} {'sinusoidal':>12} {'learned PE':>12}")
    for r in rows:
        print(f"{r['M']:>3} {r['sinusoidal']:>12.4f} {r['learned']:>12.4f}")
    return rows


if __name__ == "__main__":
    main()
