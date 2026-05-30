"""Mechanistic interpretability of the 2-layer pointer transformer.

Trains a 2-layer, 1-head transformer on the M=8 pointer-dereferencing
task (the post-6 baseline), then reverse-engineers the circuit:

  Layer 1: aggregates address bits a_0, a_1, a_2 at position 10
           (the last input position), since the embedding there only
           sees a_2 directly.
  Layer 2: uses the assembled address a in {0,...,7} to attend to
           memory position m_a and copies its value.

Produces:
  - Attention heatmaps for representative examples (layer 1, layer 2)
  - "Where does position 10 attend?" aggregate analysis across the
    test set, partitioned by address value
  - Ablation table: zero-out layer 1 attn, zero-out layer 2 attn,
    swap layer 2 address with random bits
  - QK and OV norm summaries (Anthropic transformer-circuits frame)

Run: python examples/pointer_interp.py
"""
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointer_experiments import train_transformer_on_examples, eval_transformer
from pointer_transformer import BitTransformer, count_params
from simple_pointer_dgp import make_variant1


FIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "docs", "series", "figures", "07-interp",
)


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def address_value(ids, M=8, A=3):
    """Decode the address bits (positions M, M+1, ..., M+A-1) to an int."""
    bits = ids[M:M + A]
    a = 0
    for b in bits:
        a = (a << 1) | int(b)
    return a


def attention_for_example(model, ids):
    """Return (attn_L1, attn_L2) for a single 1-head 2-layer model."""
    attn_L1 = model.attention_at(ids, layer=0, head=0)
    attn_L2 = model.attention_at(ids, layer=1, head=0)
    return attn_L1, attn_L2


def plot_attention_heatmap(attn, title, out_path, M=8, A=3):
    T = attn.shape[0]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1.0, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = list(range(T))
    labels = [f"m{i}" for i in range(M)] + [f"a{i}" for i in range(A)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Key position (attended-to)")
    ax.set_ylabel("Query position")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def aggregate_last_row(model, X, M=8, A=3):
    """For each test example, return the layer-2 attention weights from
    position M+A-1 (the last input position) to all earlier positions.
    Group by the encoded address value a in {0, ..., 2^A - 1}.
    """
    T = M + A
    by_addr = {a: [] for a in range(2 ** A)}
    for ids in X:
        ctx = ids[:T]
        a = address_value(ctx, M=M, A=A)
        attn_L2 = model.attention_at(ctx, layer=1, head=0)
        last_row = attn_L2[T - 1]
        by_addr[a].append(last_row)
    means = {a: np.mean(np.stack(v), axis=0) for a, v in by_addr.items() if v}
    return means


def plot_addr_partition(means, out_path, M=8, A=3):
    T = M + A
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for a, ax in zip(sorted(means.keys()), axes.flat):
        row = means[a]
        colors = ["#888"] * T
        colors[a] = "#d62728"
        ax.bar(range(T), row, color=colors)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"address a = {a}  (red = m{a})", fontsize=10)
        ax.set_xticks(range(T))
        labels = [f"m{i}" for i in range(M)] + [f"a{i}" for i in range(A)]
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Layer-2 attention from position 10 (last token), "
                 "averaged across test examples with each address", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def ablation_zero_layer_attn(model, X, Y, layer_idx, M=8, A=3):
    """Replace the cached attention pattern for a given layer with the
    uniform-on-causal-prefix distribution before computing the prediction.
    Implemented by monkey-patching the forward of that block.

    Cleanest path: rerun forward, then manually substitute attn at the
    chosen layer with uniform and recompute downstream. Since this is a
    small model we just clone numpy state and recompute by hand.
    """
    correct = 0
    T = M + A
    uniform = np.tril(np.ones((T, T), dtype=np.float32))
    uniform = uniform / uniform.sum(axis=-1, keepdims=True)

    for ids, y in zip(X, Y):
        ctx = ids[:T]
        x = model.embed.forward(ctx) + model.pe[:T]

        for li, blk in enumerate(model.blocks):
            if li != layer_idx:
                x = blk.forward(x)
                continue
            # Manually run this block with the attention replaced by uniform.
            # Block: x = x + attn(layer_norm(x)); x = x + ffn(layer_norm(x))
            x_pre = blk.ln1.forward(x)
            T_now, D = x_pre.shape
            H, Dh = blk.attn.n_heads, blk.attn.head_dim
            q = blk.attn.W_q.forward(x_pre)
            k = blk.attn.W_k.forward(x_pre)
            v = blk.attn.W_v.forward(x_pre)
            Q = q.reshape(T_now, H, Dh).transpose(1, 0, 2)
            K = k.reshape(T_now, H, Dh).transpose(1, 0, 2)
            V = v.reshape(T_now, H, Dh).transpose(1, 0, 2)
            attn_pat = np.broadcast_to(uniform[None], (H, T_now, T_now))
            heads_out = attn_pat @ V
            merged = heads_out.transpose(1, 0, 2).reshape(T_now, D)
            attn_out = blk.attn.W_o.forward(merged)
            x_after_attn = x + attn_out
            x = x_after_attn + blk.ffn.forward(blk.ln2.forward(x_after_attn))

        logits = model.head.forward(model.ln_final.forward(x))
        pred = int(np.argmax(logits[-1]))
        if pred == int(y):
            correct += 1
    return correct / len(X)


def ablation_shuffle_address(model, X, Y, M=8, A=3, seed=0):
    """Permute the address bits across each example before forward.
    The model should fail since it cannot recover the original address."""
    rng = np.random.default_rng(seed)
    correct = 0
    T = M + A
    for ids, y in zip(X, Y):
        ctx = ids[:T].copy()
        addr = ctx[M:M + A].copy()
        rng.shuffle(addr)
        ctx[M:M + A] = addr
        logits = model.forward(ctx)
        pred = int(np.argmax(logits[-1]))
        if pred == int(y):
            correct += 1
    return correct / len(X)


def main(M=8, A=3, n_iters=2000, batch_size=32, lr=1e-3, seed=0):
    ensure_fig_dir()
    print(f"=== Pointer interpretability run ===")
    print(f"M={M}, A={A}, T={M + A}")

    X_tr, Y_tr = make_variant1(20000, M=M, A=A, seed=seed)
    X_te, Y_te = make_variant1(2000, M=M, A=A, seed=seed + 1)

    model = BitTransformer(d_model=32, n_heads=1, d_ff=64, n_layers=2,
                            max_T=M + A, seed=seed)
    print(f"parameters: {count_params(model):,}")

    print(f"\nTraining {n_iters} iters at lr={lr} ...")
    train_transformer_on_examples(model, X_tr, Y_tr, n_iters=n_iters, lr=lr,
                       batch_size=batch_size, log_every=500, seed=seed)
    acc, loss = eval_transformer(model, X_te, Y_te)
    print(f"Test accuracy: {acc:.4f}  loss: {loss:.4f}")
    if acc < 0.99:
        print("WARNING: model did not converge to >=99% accuracy; "
              "interpretation may be on a non-solving model.")

    # === Attention heatmaps for representative examples ===
    print("\n=== Per-example attention patterns ===")
    representative = []
    seen_addrs = set()
    for ids in X_te:
        a = address_value(ids, M=M, A=A)
        if a in seen_addrs:
            continue
        representative.append((a, ids[:M + A].copy()))
        seen_addrs.add(a)
        if len(seen_addrs) == min(4, 2 ** A):
            break

    for a, ctx in representative:
        attn_L1, attn_L2 = attention_for_example(model, ctx)
        mem_bits = "".join(str(int(b)) for b in ctx[:M])
        addr_bits = "".join(str(int(b)) for b in ctx[M:M + A])
        target = int(ctx[a])
        title_L1 = (f"Layer 1 attn  |  mem={mem_bits}  addr={addr_bits} "
                    f"(a={a})  target={target}")
        title_L2 = (f"Layer 2 attn  |  mem={mem_bits}  addr={addr_bits} "
                    f"(a={a})  target={target}")
        plot_attention_heatmap(
            attn_L1, title_L1,
            os.path.join(FIG_DIR, f"attn_L1_a{a}.png"), M=M, A=A,
        )
        plot_attention_heatmap(
            attn_L2, title_L2,
            os.path.join(FIG_DIR, f"attn_L2_a{a}.png"), M=M, A=A,
        )
        L2_last = attn_L2[M + A - 1]
        L2_argmax = int(np.argmax(L2_last))
        L2_w = float(L2_last[L2_argmax])
        L1_last = attn_L1[M + A - 1]
        L1_top_pos = list(np.argsort(L1_last)[::-1][:3])
        L1_top_w = [float(L1_last[p]) for p in L1_top_pos]
        addr_positions = list(range(M, M + A))
        L1_on_addr = float(L1_last[addr_positions].sum())
        L1_on_mem = float(L1_last[:M].sum())
        print(f"  a={a}: L2 last-row argmax = pos {L2_argmax} "
              f"(weight {L2_w:.3f}, expected m_{a});  "
              f"L1 last-row top-3 = {list(zip(L1_top_pos, [f'{w:.2f}' for w in L1_top_w]))};  "
              f"L1 on addr positions = {L1_on_addr:.3f}, on memory = {L1_on_mem:.3f}")

    # === Aggregate: average layer-2 last-row by address ===
    print("\n=== Aggregate last-row layer-2 attention by address ===")
    means = aggregate_last_row(model, X_te[:500], M=M, A=A)
    plot_addr_partition(means, os.path.join(FIG_DIR, "addr_partition.png"),
                        M=M, A=A)
    for a in sorted(means.keys()):
        row = means[a]
        target = a
        on_target = float(row[target])
        other = float(row[:M].sum() - row[target])
        print(f"  a={a}: average weight on m_{a} = {on_target:.3f}; "
              f"on other memory positions = {other:.3f}")

    # === Ablations ===
    print("\n=== Ablations ===")
    base_acc, _ = eval_transformer(model, X_te, Y_te)
    print(f"  baseline:                  acc {base_acc:.4f}")
    acc_zero_L1 = ablation_zero_layer_attn(model, X_te[:500], Y_te[:500],
                                            layer_idx=0, M=M, A=A)
    print(f"  layer-1 attn -> uniform:   acc {acc_zero_L1:.4f}")
    acc_zero_L2 = ablation_zero_layer_attn(model, X_te[:500], Y_te[:500],
                                            layer_idx=1, M=M, A=A)
    print(f"  layer-2 attn -> uniform:   acc {acc_zero_L2:.4f}")
    acc_shuffle = ablation_shuffle_address(model, X_te[:500], Y_te[:500],
                                            M=M, A=A, seed=seed + 7)
    print(f"  shuffle address bits:      acc {acc_shuffle:.4f}")

    # === QK and OV matrix norms (per layer) ===
    print("\n=== QK and OV matrix scale per layer ===")
    for li, blk in enumerate(model.blocks):
        Wq = blk.attn.W_q.W
        Wk = blk.attn.W_k.W
        Wv = blk.attn.W_v.W
        Wo = blk.attn.W_o.W
        QK = Wq @ Wk.T
        OV = Wo @ Wv  # Wo is (D, D); Wv is (D, D); OV: (D, D)
        print(f"  layer {li + 1}: ||W_q||={np.linalg.norm(Wq):.2f} "
              f"||W_k||={np.linalg.norm(Wk):.2f} "
              f"||W_v||={np.linalg.norm(Wv):.2f} "
              f"||W_o||={np.linalg.norm(Wo):.2f} "
              f"||QK||={np.linalg.norm(QK):.2f} "
              f"||OV||={np.linalg.norm(OV):.2f}")

    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
