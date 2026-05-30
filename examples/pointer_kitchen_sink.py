"""Try the 'production transformer recipe' on the higher-M lookup task.

Section 10 of transformer-pointers.md found that learned PE solves
M=16 (100%) but plateaus at M>=24. The hypothesis is that an
additional bottleneck kicks in once the address has 5 or more bits,
and that the standard production-transformer recipe (multi-head +
larger d_model + LR warmup + learned PE) closes it.

We test the kitchen-sink recipe at M=24, M=32. If it cracks both,
the "necessary but not sufficient" lesson generalizes cleanly. If
even the kitchen sink fails, the small-NumPy regime has fundamental
limits we cannot escape.

Recipe:
  - Learned PE (small-init, same scale as embedding)
  - d_model = 128 (vs 64 in the failed runs)
  - n_heads = 4 (vs 1)
  - n_layers = 2 (unchanged)
  - LR = 1e-3 with linear warmup over first 500 iters
  - batch = 32
  - 8000 iters (vs 6000)

Run: python examples/pointer_kitchen_sink.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_pointer_dgp import make_variant1
from pointer_transformer import count_params
from pointer_experiments import eval_transformer
from pointer_learned_pe import LearnedPETransformer
from transformer import softmax_cross_entropy, Adam


class WarmupAdam:
    """Adam with linear LR warmup over the first `warmup_iters` calls."""

    def __init__(self, params, peak_lr=1e-3, warmup_iters=500,
                 betas=(0.9, 0.95), eps=1e-8):
        self.peak_lr = peak_lr
        self.warmup_iters = warmup_iters
        self.adam = Adam(params, lr=0.0, betas=betas, eps=eps)
        self.t = 0

    def step(self):
        self.t += 1
        if self.t <= self.warmup_iters:
            self.adam.lr = self.peak_lr * (self.t / self.warmup_iters)
        else:
            self.adam.lr = self.peak_lr
        self.adam.step()


def train_kitchen_sink(model, X, Y, n_iters, peak_lr=1e-3, warmup=500,
                       batch_size=32, log_every=1000, seed=0):
    opt = WarmupAdam(model.params(), peak_lr=peak_lr, warmup_iters=warmup)
    rng = np.random.default_rng(seed)
    running = 0.0
    t0 = time.time()
    for it in range(1, n_iters + 1):
        idx = rng.integers(0, len(X), size=batch_size)
        model.zero_grad()
        total = 0.0
        for b in idx:
            ctx = X[b][:-1]
            target = int(X[b][-1])
            logits = model.forward(ctx)
            last = logits[-1]
            mx = last.max()
            e = np.exp(last - mx)
            z = float(e.sum())
            total += float(mx - last[target] + math.log(z))
            probs = e / z
            grad = probs.copy()
            grad[target] -= 1.0
            dlogits = np.zeros_like(logits)
            dlogits[-1] = grad / batch_size
            model.backward(dlogits)
        opt.step()
        running += total / batch_size
        if it % log_every == 0:
            avg = running / log_every
            print(f"    iter {it:5d}  loss {avg:.4f}  "
                  f"lr {opt.adam.lr:.5f}  ({time.time()-t0:.0f}s)")
            running = 0.0


def main(M_values=(16, 24, 32), n_iters=8000, peak_lr=1e-3, warmup=500,
         seed=0):
    print(f"Kitchen-sink recipe: d_model=128, n_heads=4, learned PE, "
          f"warmup={warmup}, n_iters={n_iters}\n")
    for M in M_values:
        A = max(3, (M - 1).bit_length())
        T = M + A
        print(f"{'-' * 60}\nM={M}, A={A}, T={T}\n{'-' * 60}")
        X_tr, Y_tr = make_variant1(20000, M=M, A=A, seed=seed)
        X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1)

        m = LearnedPETransformer(d_model=128, n_heads=4, d_ff=256,
                                  n_layers=2, max_T=T, seed=seed)
        print(f"  parameters: {count_params(m):,}")
        train_kitchen_sink(m, X_tr, Y_tr, n_iters=n_iters,
                            peak_lr=peak_lr, warmup=warmup,
                            batch_size=32, log_every=1000, seed=seed)
        acc, loss = eval_transformer(m, X_te, Y_te)
        print(f"  test acc {acc:.4f}  loss {loss:.4f}")


if __name__ == "__main__":
    main()
