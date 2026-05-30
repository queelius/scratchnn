"""PyTorch scaling sweep for the M=32 pointer-dereferencing task.

Follows the NumPy kitchen-sink runs (in examples/) that established
the existing M=16, M=24, M=32 results. PyTorch is the right tool for
*scaling* the recipe; the NumPy implementation in examples/ remains
the pedagogical anchor (from-scratch backprop, all math visible).

Configs tested (all on M=32, A=5, T=37, batch=32, 15000 iters, Adam
lr=1e-3 with linear warmup over first 500 iters):

  1. baseline match    d_model=128, n_heads=4, n_layers=2
                       (matches NumPy kitchen-sink; sanity check)
  2. 2x width          d_model=256, n_heads=8, n_layers=2
  3. depth             d_model=128, n_heads=4, n_layers=3
  4. wide + deep       d_model=256, n_heads=8, n_layers=3

Existing M=32 results (NumPy, for comparison only, not rerun here):
  - baseline transformer (sinusoidal PE):              acc 0.664
  - learned PE alone:                                   acc 0.645
  - kitchen-sink (d=128, h=4, 8000 iters):              acc 0.680
  - kitchen-sink long (d=128, h=4, 30000 iters):        acc 0.7625
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simple_pointer_dgp import make_variant1


class PointerTransformer(nn.Module):
    """Decoder-only transformer over the 2-token bit vocabulary.

    PreLN blocks via nn.TransformerEncoderLayer with norm_first=True
    and a causal mask. Learned positional embedding at small init
    scale (the post-6 lesson on PE-scale mismatch).

    No dropout, no batch norm, so we never need to toggle train/eval
    modes; the model behaves identically either way.
    """

    def __init__(self, d_model, n_heads, n_layers, d_ff, max_T, vocab=2):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_T, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            activation="gelu", batch_first=True, norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.max_T = max_T

    def forward(self, ids):
        # ids: (B, T)
        T = ids.size(1)
        pos_ids = torch.arange(T, device=ids.device)
        x = self.embed(ids) + self.pos(pos_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_config(name, d_model, n_heads, n_layers, d_ff, M, A, T,
               X_tr, Y_tr, X_te, Y_te,
               n_iters=15000, peak_lr=1e-3, warmup=500, batch_size=32,
               seed=0, log_every=1000):
    torch.manual_seed(seed)
    model = PointerTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, max_T=T,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.95))

    print(f"\n{'-' * 60}\n[{name}]  d_model={d_model}, n_heads={n_heads}, "
          f"n_layers={n_layers}, d_ff={d_ff}\n{'-' * 60}")
    print(f"  parameters: {count_params(model):,}")

    rng = np.random.default_rng(seed)
    X_tr_t = torch.from_numpy(X_tr).long()
    X_te_t = torch.from_numpy(X_te).long()

    t0 = time.time()
    running = 0.0
    for it in range(1, n_iters + 1):
        idx = rng.integers(0, len(X_tr), size=batch_size)
        idx_t = torch.from_numpy(idx).long()
        ctx = X_tr_t[idx_t, :-1]                 # (B, T)
        target = X_tr_t[idx_t, -1]               # (B,)
        logits = model(ctx)                      # (B, T, vocab)
        last = logits[:, -1, :]                  # (B, vocab)
        loss = F.cross_entropy(last, target)

        opt.zero_grad()
        loss.backward()
        if it <= warmup:
            lr_now = peak_lr * it / warmup
        else:
            lr_now = peak_lr
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.step()
        running += float(loss.item())

        if it % log_every == 0:
            avg = running / log_every
            print(f"    iter {it:5d}  loss {avg:.4f}  lr {lr_now:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            running = 0.0

    with torch.no_grad():
        correct = 0
        total_loss = 0.0
        N = len(X_te)
        bsz = 256
        for i in range(0, N, bsz):
            ctx = X_te_t[i:i + bsz, :-1]
            tgt = X_te_t[i:i + bsz, -1]
            logits = model(ctx)
            last = logits[:, -1, :]
            pred = last.argmax(dim=-1)
            correct += int((pred == tgt).sum().item())
            total_loss += float(F.cross_entropy(last, tgt,
                                                reduction="sum").item())
    acc = correct / N
    avg_loss = total_loss / N
    elapsed = time.time() - t0
    print(f"  test acc {acc:.4f}  loss {avg_loss:.4f}  (total {elapsed:.0f}s)",
          flush=True)
    return {
        "name": name, "d_model": d_model, "n_heads": n_heads,
        "n_layers": n_layers, "d_ff": d_ff, "params": count_params(model),
        "n_iters": n_iters, "test_acc": acc, "test_loss": avg_loss,
        "elapsed_s": elapsed,
    }


def main(M=32, n_iters=15000, seed=0):
    A = max(3, (M - 1).bit_length())
    T = M + A
    print(f"M={M}, A={A}, T={T}, batch=32, lr=1e-3, warmup=500")

    X_tr, Y_tr = make_variant1(20000, M=M, A=A, seed=seed)
    X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1)

    configs = [
        ("baseline-match", 128, 4, 2, 256),
        ("2x-width",       256, 8, 2, 512),
        ("3L-depth",       128, 4, 3, 256),
        ("wide+deep",      256, 8, 3, 512),
    ]
    results = []
    for name, d_model, n_heads, n_layers, d_ff in configs:
        r = run_config(name, d_model, n_heads, n_layers, d_ff, M, A, T,
                       X_tr, Y_tr, X_te, Y_te,
                       n_iters=n_iters, seed=seed)
        results.append(r)

    print(f"\n{'=' * 70}\nSummary on M={M} (PyTorch CPU)\n{'=' * 70}")
    print(f"{'config':<20} {'params':>10} {'acc':>8} {'loss':>8} {'time':>8}")
    for r in results:
        print(f"{r['name']:<20} {r['params']:>10,} "
              f"{r['test_acc']:>8.4f} {r['test_loss']:>8.4f} "
              f"{r['elapsed_s']:>7.0f}s")


if __name__ == "__main__":
    main()
