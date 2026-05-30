"""Clean depth comparison for the M=32 pointer task (PyTorch).

The 4-config sweep (pointer_sweep.py) found that depth, not width, is
the M=32 knob: 3 layers reached 0.78 with a phase transition while
doubling width at 2 layers stayed at chance-plus. But that comparison
confounds depth with parameter count and convergence speed.

This script isolates depth. It holds d_model=128, n_heads=4, d_ff=256
fixed and varies only n_layers, at a single extended iteration budget.
Run it once per layer count:

    python examples/pytorch/pointer_depth.py --layers 2 --iters 40000
    python examples/pytorch/pointer_depth.py --layers 3 --iters 40000

If 3L solves M=32 and 2L plateaus at the same budget, depth is the
knob, cleanly. Logs every 1000 iters so the phase transition (if any)
is visible.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from simple_pointer_dgp import make_variant1
from pointer_sweep import run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    M = args.M
    A = max(3, (M - 1).bit_length())
    T = M + A
    print(f"Depth comparison: layers={args.layers}, heads={args.heads}, "
          f"d_model={args.d_model}, d_ff={args.d_ff}")
    print(f"M={M}, A={A}, T={T}, iters={args.iters}, seed={args.seed}")

    X_tr, Y_tr = make_variant1(20000, M=M, A=A, seed=args.seed)
    X_te, Y_te = make_variant1(2000, M=M, A=A, seed=args.seed + 1)

    run_config(
        name=f"{args.layers}L-extended",
        d_model=args.d_model, n_heads=args.heads,
        n_layers=args.layers, d_ff=args.d_ff,
        M=M, A=A, T=T,
        X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te,
        n_iters=args.iters, seed=args.seed, log_every=1000,
    )


if __name__ == "__main__":
    main()
