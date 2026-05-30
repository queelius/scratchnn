"""Extra-large kitchen-sink recipe on M=32.

The 30000-iter long run at kitchen-sink width (d_model=128, n_heads=4)
got M=32 to test acc 0.7625, with the loss descending slowly (0.34 at
the end, no phase transition). Section 10 of the transformer-pointers
post argued that wider or deeper might be the missing knob. This
script doubles the width and the head count to see whether width is
the bottleneck at M=32.

Recipe:
  - d_model = 256 (vs 128 in kitchen-sink)
  - n_heads = 8 (vs 4)
  - d_ff = 512 (vs 256)
  - n_layers = 2 (unchanged)
  - Learned PE at small init scale
  - LR = 1e-3 with linear warmup over first 500 iters
  - batch = 32
  - 15000 iters
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
from pointer_kitchen_sink import WarmupAdam, train_kitchen_sink


def main(M=32, n_iters=15000, peak_lr=1e-3, warmup=500, seed=0):
    A = max(3, (M - 1).bit_length())
    T = M + A
    print(f"Extra-large kitchen-sink: d_model=256, n_heads=8, "
          f"learned PE, warmup={warmup}, n_iters={n_iters}\n")
    print(f"{'-' * 60}\nM={M}, A={A}, T={T}\n{'-' * 60}")
    X_tr, Y_tr = make_variant1(20000, M=M, A=A, seed=seed)
    X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1)
    m = LearnedPETransformer(d_model=256, n_heads=8, d_ff=512,
                              n_layers=2, max_T=T, seed=seed)
    print(f"  parameters: {count_params(m):,}")
    train_kitchen_sink(m, X_tr, Y_tr, n_iters=n_iters,
                        peak_lr=peak_lr, warmup=warmup,
                        batch_size=32, log_every=1000, seed=seed)
    acc, loss = eval_transformer(m, X_te, Y_te)
    print(f"  test acc {acc:.4f}  loss {loss:.4f}")


if __name__ == "__main__":
    main()
