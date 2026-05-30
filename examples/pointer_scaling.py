"""Scaling experiment: do MLPs and transformers behave differently as the
memory size M grows?

The claim from the transformer-pointers post is that an MLP succeeds on
the single-lookup task at small M only because it can memorize the
input-output table (2^(M+A) patterns total). As M grows, the input space
grows exponentially while the training data stays fixed, so memorization
becomes impossible. The transformer's content-addressable-memory
mechanism has parameter count O(M*d_model^2), which scales linearly in
M, so it should keep working.

This experiment scales M across {8, 16, 24, 32} and trains:

  - An MLP with hidden=128 (high capacity, dwarfs the transformer)
  - A 2-layer 1-head transformer with d_model=32

Both use 20k training examples and 3000 Adam iterations. We expect MLP
accuracy to degrade with M while transformer accuracy stays high.

Run: python examples/pointer_scaling.py
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


def addr_bits_for(M):
    """Smallest A with 2**A >= M. Min 3 to keep things uniform."""
    return max(3, (M - 1).bit_length())


def run_scaling(M_values, n_train=20000, n_test=2000, n_iters=3000,
                mlp_hidden=128, batch_size=32, seed=0):
    rows = []
    for M in M_values:
        A = addr_bits_for(M)
        T = M + A  # context length (target is the +1, not in context)
        print(f"\n{'=' * 70}")
        print(f"M = {M}, A = {A}, sequence length = {M + A + 1}, "
              f"input space size = 2^{M + A} = {2 ** (M + A)}")
        print(f"  data: {n_train} train, {n_test} test ({100 * n_train / 2 ** (M+A):.3f}% of input space)")
        print(f"{'=' * 70}")

        X_tr, Y_tr = make_variant1(n_train, M=M, A=A, seed=seed)
        X_te, Y_te = make_variant1(n_test, M=M, A=A, seed=seed + 1)

        # MLP: hidden=128 across all M, so capacity is constant-and-large.
        mlp = MLPBaseline(context_len=T, embed_dim=4, hidden=mlp_hidden,
                          seed=seed)
        n_mlp = count_params(mlp)
        print(f"\n  MLP (hidden={mlp_hidden}):")
        print(f"    parameters: {n_mlp:,}")
        t0 = time.time()
        train_mlp_on_examples(mlp, X_tr, Y_tr, n_iters=n_iters,
                              batch_size=64, lr=1e-3, log_every=500,
                              seed=seed, silent=True)
        mlp_acc, mlp_loss = eval_mlp(mlp, X_te, Y_te)
        print(f"    test acc {mlp_acc:.4f}  loss {mlp_loss:.4f}  "
              f"(train time {time.time() - t0:.0f}s)")

        # 2-layer 1-head transformer: same architecture across all M, only
        # max_T changes.
        xf = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=2,
                            max_T=T, seed=seed)
        n_xf = count_params(xf)
        print(f"\n  Transformer (2 layers, 1 head, d_model=32):")
        print(f"    parameters: {n_xf:,}")
        t0 = time.time()
        train_transformer_on_examples(xf, X_tr, Y_tr, n_iters=n_iters,
                                       batch_size=batch_size, lr=1e-3,
                                       log_every=500, seed=seed,
                                       silent=True)
        xf_acc, xf_loss = eval_transformer(xf, X_te, Y_te)
        print(f"    test acc {xf_acc:.4f}  loss {xf_loss:.4f}  "
              f"(train time {time.time() - t0:.0f}s)")

        rows.append({
            'M': M, 'A': A, 'T': T,
            'input_space': 2 ** (M + A),
            'mlp_params': n_mlp, 'mlp_acc': mlp_acc, 'mlp_loss': mlp_loss,
            'xf_params': n_xf, 'xf_acc': xf_acc, 'xf_loss': xf_loss,
        })

    # Summary
    print(f"\n{'=' * 70}\nSCALING SUMMARY\n{'=' * 70}")
    print(f"{'M':>4} {'A':>3} {'2^(M+A)':>14} {'MLP params':>12} "
          f"{'MLP acc':>9} {'XF params':>11} {'XF acc':>9}")
    print("-" * 70)
    for r in rows:
        print(f"{r['M']:>4} {r['A']:>3} {r['input_space']:>14,} "
              f"{r['mlp_params']:>12,} {r['mlp_acc']:>9.4f} "
              f"{r['xf_params']:>11,} {r['xf_acc']:>9.4f}")
    return rows


if __name__ == "__main__":
    run_scaling(M_values=[8, 16, 24, 32], n_iters=3000)
