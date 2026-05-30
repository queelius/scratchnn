"""Transformer experiments on the simple memory + pointer DGP.

Trains and evaluates models on the three simple-pointer task variants
from `simple_pointer_dgp.py`:

  Variant 1 (single lookup):
      [memory M=8] [address A=3] [target = memory[addr]]
      Shows the canonical 1-layer-fails / 2-layer-succeeds depth gap.

  Variant 2 (two lookups + XOR):
      [memory M=8] [addr1 A=3] [addr2 A=3] [target = m[a1] XOR m[a2]]
      Shows that 1-head struggles where 2-heads specialize cleanly.

  Variant 3 (pointer to pointer):
      [memory M=8] [addr A=3] [target = memory[memory[addr] ...]]
      Shows depth scaling: needs more layers for more hops.

Each model is the small NumPy transformer from
`examples/pointer_transformer.py`, sized to the variant.

Run: python examples/pointer_experiments.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_pointer_dgp import make_variant1, make_variant2, make_variant3
from pointer_transformer import BitTransformer, MLPBaseline, count_params
from transformer import Adam


# ============================================================
# Per-example training
# ============================================================

def train_transformer_on_examples(model, X, Y, n_iters=2000, batch_size=32,
                                   lr=1e-3, log_every=200, seed=0,
                                   silent=False):
    """Train on the fixed-format pointer task. X[b] is a length-L sequence,
    target is X[b][-1] given X[b][:-1]."""
    optimizer = Adam(model.params(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
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
        optimizer.step()
        running += total / batch_size
        if it % log_every == 0:
            avg = running / log_every
            history.append(avg)
            if not silent:
                print(f"      iter {it:4d}  loss {avg:.4f}  ({time.time()-t0:.0f}s)")
            running = 0.0
    return history


def eval_transformer(model, X, Y):
    correct = 0
    total_loss = 0.0
    for i in range(len(X)):
        ctx = X[i][:-1]
        target = int(Y[i])
        logits = model.forward(ctx)
        last = logits[-1]
        mx = last.max()
        e = np.exp(last - mx)
        z = float(e.sum())
        total_loss += float(mx - last[target] + math.log(z))
        if int(np.argmax(last)) == target:
            correct += 1
    return correct / len(X), total_loss / len(X)


def train_mlp_on_examples(model, X, Y, n_iters=2000, batch_size=64, lr=1e-3,
                          log_every=200, seed=0, silent=False):
    optimizer = Adam(model.params(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    running = 0.0
    t0 = time.time()
    for it in range(1, n_iters + 1):
        idx = rng.integers(0, len(X), size=batch_size)
        model.zero_grad()
        total = 0.0
        for b in idx:
            ctx = X[b][:-1]
            target = int(X[b][-1])
            logits = model.forward_one(ctx)
            mx = logits.max()
            e = np.exp(logits - mx)
            z = float(e.sum())
            loss = float(mx - logits[target] + math.log(z))
            total += loss
            probs = e / z
            grad = probs.copy()
            grad[target] -= 1.0
            grad /= batch_size
            model.backward_one(grad)
        optimizer.step()
        running += total / batch_size
        if it % log_every == 0:
            avg = running / log_every
            history.append(avg)
            if not silent:
                print(f"      iter {it:4d}  loss {avg:.4f}  ({time.time()-t0:.0f}s)")
            running = 0.0
    return history


def eval_mlp(model, X, Y):
    correct = 0
    total_loss = 0.0
    for i in range(len(X)):
        ctx = X[i][:-1]
        target = int(Y[i])
        logits = model.forward_one(ctx)
        mx = logits.max()
        e = np.exp(logits - mx)
        z = float(e.sum())
        total_loss += float(mx - logits[target] + math.log(z))
        if int(np.argmax(logits)) == target:
            correct += 1
    return correct / len(X), total_loss / len(X)


# ============================================================
# Experiments
# ============================================================

def experiment_1(seed=0):
    """Variant 1: single lookup. Shows depth gap."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Single lookup (Variant 1)")
    print("  Task: [memory M=8] [address A=3] [target = memory[addr]]")
    print("=" * 70)

    X_tr, Y_tr = make_variant1(20000, M=8, A=3, seed=seed)
    X_te, Y_te = make_variant1(2000, M=8, A=3, seed=seed + 1)
    L_ctx = X_tr.shape[1] - 1  # context length

    results = {}

    print("\n  [a] MLP baseline (matched params)")
    mlp = MLPBaseline(context_len=L_ctx, embed_dim=4, hidden=64, seed=seed)
    print(f"      params: {count_params(mlp)}")
    train_mlp_on_examples(mlp, X_tr, Y_tr, n_iters=2000, lr=1e-3, seed=seed)
    acc, loss = eval_mlp(mlp, X_te, Y_te)
    print(f"      test acc {acc:.4f}  loss {loss:.4f}")
    results['mlp'] = (acc, loss)

    print("\n  [b] 1-layer 1-head transformer")
    t11 = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=1,
                         max_T=L_ctx, seed=seed)
    print(f"      params: {count_params(t11)}")
    train_transformer_on_examples(t11, X_tr, Y_tr, n_iters=2000, lr=1e-3,
                                   seed=seed)
    acc, loss = eval_transformer(t11, X_te, Y_te)
    print(f"      test acc {acc:.4f}  loss {loss:.4f}")
    results['t11'] = (acc, loss)

    print("\n  [c] 2-layer 1-head transformer")
    t21 = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=2,
                         max_T=L_ctx, seed=seed)
    print(f"      params: {count_params(t21)}")
    train_transformer_on_examples(t21, X_tr, Y_tr, n_iters=2000, lr=1e-3,
                                   seed=seed)
    acc, loss = eval_transformer(t21, X_te, Y_te)
    print(f"      test acc {acc:.4f}  loss {loss:.4f}")
    results['t21'] = (acc, loss)

    print("\n  Summary (Variant 1):")
    print(f"    MLP                : acc {results['mlp'][0]:.3f}  loss {results['mlp'][1]:.4f}")
    print(f"    Transformer (1L,1H): acc {results['t11'][0]:.3f}  loss {results['t11'][1]:.4f}")
    print(f"    Transformer (2L,1H): acc {results['t21'][0]:.3f}  loss {results['t21'][1]:.4f}")
    return results


def experiment_2(seed=0):
    """Variant 2: two lookups + XOR. Shows multi-head specialization."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Two lookups + XOR (Variant 2)")
    print("  Task: [memory M=8] [a1 A=3] [a2 A=3] [t = m[a1] XOR m[a2]]")
    print("=" * 70)

    X_tr, Y_tr = make_variant2(30000, M=8, A=3, seed=seed)
    X_te, Y_te = make_variant2(3000, M=8, A=3, seed=seed + 1)
    L_ctx = X_tr.shape[1] - 1

    results = {}

    print("\n  [a] 2-layer 1-head transformer")
    t21 = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=2,
                         max_T=L_ctx, seed=seed)
    print(f"      params: {count_params(t21)}")
    train_transformer_on_examples(t21, X_tr, Y_tr, n_iters=3000, lr=1e-3,
                                   seed=seed)
    acc, loss = eval_transformer(t21, X_te, Y_te)
    print(f"      test acc {acc:.4f}  loss {loss:.4f}")
    results['1H'] = (acc, loss)

    print("\n  [b] 2-layer 2-head transformer")
    t22 = BitTransformer(d_model=32, n_heads=2, d_ff=64, n_layers=2,
                         max_T=L_ctx, seed=seed)
    print(f"      params: {count_params(t22)}")
    train_transformer_on_examples(t22, X_tr, Y_tr, n_iters=3000, lr=1e-3,
                                   seed=seed)
    acc, loss = eval_transformer(t22, X_te, Y_te)
    print(f"      test acc {acc:.4f}  loss {loss:.4f}")
    results['2H'] = (acc, loss)

    print("\n  Summary (Variant 2):")
    print(f"    Transformer (2L,1H): acc {results['1H'][0]:.3f}  loss {results['1H'][1]:.4f}")
    print(f"    Transformer (2L,2H): acc {results['2H'][0]:.3f}  loss {results['2H'][1]:.4f}")
    return results


def experiment_3(seed=0):
    """Variant 3: pointer to pointer. Shows depth scaling for multi-hop."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Pointer to pointer (Variant 3)")
    print("  Task: [memory M=8] [addr A=3] [target = m[m[addr] as new addr]]")
    print("=" * 70)

    X_tr, Y_tr = make_variant3(20000, M=8, A=3, seed=seed)
    X_te, Y_te = make_variant3(2000, M=8, A=3, seed=seed + 1)
    L_ctx = X_tr.shape[1] - 1

    results = {}

    print("\n  [a] 2-layer 1-head transformer")
    t21 = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=2,
                         max_T=L_ctx, seed=seed)
    print(f"      params: {count_params(t21)}")
    train_transformer_on_examples(t21, X_tr, Y_tr, n_iters=2500, lr=1e-3,
                                   seed=seed)
    acc, loss = eval_transformer(t21, X_te, Y_te)
    print(f"      test acc {acc:.4f}  loss {loss:.4f}")
    results['2L'] = (acc, loss)

    print("\n  [b] 3-layer 1-head transformer")
    t31 = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=3,
                         max_T=L_ctx, seed=seed)
    print(f"      params: {count_params(t31)}")
    train_transformer_on_examples(t31, X_tr, Y_tr, n_iters=2500, lr=1e-3,
                                   seed=seed)
    acc, loss = eval_transformer(t31, X_te, Y_te)
    print(f"      test acc {acc:.4f}  loss {loss:.4f}")
    results['3L'] = (acc, loss)

    print("\n  Summary (Variant 3):")
    print(f"    Transformer (2L,1H): acc {results['2L'][0]:.3f}  loss {results['2L'][1]:.4f}")
    print(f"    Transformer (3L,1H): acc {results['3L'][0]:.3f}  loss {results['3L'][1]:.4f}")
    return results


def main(seed=0):
    print("=" * 70)
    print("Pointer Transformer Experiments")
    print("=" * 70)
    r1 = experiment_1(seed)
    r2 = experiment_2(seed)
    r3 = experiment_3(seed)

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"\n  V1 (single lookup, 12-bit example):")
    print(f"    MLP                : acc {r1['mlp'][0]:.3f}")
    print(f"    Transformer (1L,1H): acc {r1['t11'][0]:.3f}")
    print(f"    Transformer (2L,1H): acc {r1['t21'][0]:.3f}")
    print(f"\n  V2 (two lookups + XOR, 15-bit example):")
    print(f"    Transformer (2L,1H): acc {r2['1H'][0]:.3f}")
    print(f"    Transformer (2L,2H): acc {r2['2H'][0]:.3f}")
    print(f"\n  V3 (pointer to pointer, 12-bit example):")
    print(f"    Transformer (2L,1H): acc {r3['2L'][0]:.3f}")
    print(f"    Transformer (3L,1H): acc {r3['3L'][0]:.3f}")


if __name__ == "__main__":
    main()
