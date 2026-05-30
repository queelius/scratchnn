"""Long-budget kitchen-sink run on M=32.

Section 10 of transformer-pointers.md ran the kitchen-sink recipe for
8000 iters and found M=32 still descending (loss 0.54 at the end, well
above M=24's 0.03). The natural follow-up is: how low does the loss
go with a longer budget?

This script reruns the same kitchen-sink recipe at M=32 only, for
30000 iters, to see whether the loss converges or keeps descending
at a slow rate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointer_kitchen_sink import main


if __name__ == "__main__":
    main(M_values=(32,), n_iters=30000, peak_lr=1e-3, warmup=500, seed=0)
