"""Mechanistic interpretation of the 3-layer transformer that solves M=32.

Post 7 reverse-engineered the 2-layer model at M=8: layer 1 aggregates
the address bits, layer 2 dereferences. The M=32 scaling study (post 6
section 10) found that 2 layers cannot solve M=32 no matter the width
or training time, but 3 layers can. In the writeup we hypothesized
that the third layer "buys lookup precision by inserting a refinement
stage between aggregation and final readout." This script tests that
hypothesis instead of asserting it.

To make every attention pattern directly readable, the model here is
written explicitly (multi-head causal self-attention by hand, each
layer storing its per-head attention matrix), rather than using
nn.TransformerEncoderLayer, whose attention weights are awkward to
extract. The architecture otherwise matches the controlled experiment
(d_model=128, n_heads=4, n_layers=3, learned PE).

The script:
  1. Trains the 3L model at M=32 to convergence (or loads a saved one).
  2. For each layer, measures where the last-position query attends:
     onto address positions, onto memory positions, and onto the
     addressed cell m_a (grouped by address a).
  3. Tests the refinement hypothesis: at which layer does the spike on
     m_a appear, and does it sharpen across layers?
  4. Per-head breakdown: which (layer, head) carries the lookup.
  5. Per-layer ablation (attention -> uniform) for causal confirmation.

Run: python examples/pytorch/pointer_interp_deep.py
"""
import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simple_pointer_dgp import make_variant1

FIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "docs", "series", "figures", "07-interp-deep",
)
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interp_deep_M32.pt")


# ---------------------------------------------------------------------------
# Explicit, introspectable transformer
# ---------------------------------------------------------------------------

class IntrospectableMHA(nn.Module):
    """Multi-head causal self-attention, written out so the per-head
    attention matrix is stored on every forward as self.attn (H, T, T)
    for the most recent batch element 0 (we interpret batch size 1)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn = None  # cached (H, T, T) for batch elem 0

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = self.W_q(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        k = self.W_k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, Dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,T,T)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        self.attn = attn[0].detach()  # (H, T, T) for batch elem 0
        out = attn @ v                # (B,H,T,Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = IntrospectableMHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class IntrospectableTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=3, d_ff=256,
                 max_T=37, vocab=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_T, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.max_T = max_T
        self._init_projections()

    def _init_projections(self):
        """Xavier-uniform on every Linear weight, zero biases. This matches
        the init nn.MultiheadAttention uses for its QKV projection
        (xavier_uniform_) and is what let the nn-based model solve M=32.
        Kaiming-default (a=sqrt(5)) on the attention projections mis-scales
        them and delays the phase transition past a usable budget. The
        embedding and positional table keep their small 0.02 init (the
        PE-scale lesson from post 6)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ids):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        x = self.embed(ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def layer_attn(self):
        """Return list of (H, T, T) attention tensors, one per layer,
        from the most recent forward pass (batch elem 0)."""
        return [blk.attn.attn for blk in self.blocks]

    @torch.no_grad()
    def forward_cached(self, ids):
        """Forward pass returning logits and the residual stream after the
        embedding and after each block: res[0] = embed+pos, res[L] = output
        of block L. Shapes (B, T, D)."""
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        x = self.embed(ids) + self.pos(pos)
        res = [x]
        for blk in self.blocks:
            x = blk(x)
            res.append(x)
        return self.head(self.ln_f(x)), res

    @torch.no_grad()
    def forward_patched(self, ids, layer, position, vec):
        """Forward pass that overwrites the residual at a single
        (layer, position) with `vec`. layer=0 patches the embedding output;
        layer=L patches the output of block L. Used for causal tracing."""
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        x = self.embed(ids) + self.pos(pos)
        if layer == 0:
            x = x.clone()
            x[0, position] = vec
        for li, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if li == layer:
                x = x.clone()
                x[0, position] = vec
        return self.head(self.ln_f(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Train / load
# ---------------------------------------------------------------------------

def train(model, X_tr, n_iters, peak_lr=1e-3, warmup=500, batch_size=32,
          seed=0, log_every=2000):
    opt = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(X_tr).long()
    t0_running = 0.0
    import time
    t0 = time.time()
    for it in range(1, n_iters + 1):
        idx = torch.from_numpy(rng.integers(0, len(X_tr), size=batch_size)).long()
        ctx = X[idx, :-1]
        target = X[idx, -1]
        logits = model(ctx)[:, -1, :]
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        lr_now = peak_lr * min(1.0, it / warmup)
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.step()
        t0_running += float(loss.item())
        if it % log_every == 0:
            print(f"    iter {it:5d}  loss {t0_running/log_every:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            t0_running = 0.0


@torch.no_grad()
def evaluate(model, X_te, Y_te):
    X = torch.from_numpy(X_te).long()
    correct, N, bsz = 0, len(X_te), 256
    for i in range(0, N, bsz):
        logits = model(X[i:i + bsz, :-1])[:, -1, :]
        pred = logits.argmax(-1)
        correct += int((pred == X[i:i + bsz, -1]).sum())
    return correct / N


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def address_value(ids, M, A):
    a = 0
    for b in ids[M:M + A]:
        a = (a << 1) | int(b)
    return a


@torch.no_grad()
def predict_bit(model, ctx_np):
    ids = torch.from_numpy(ctx_np[None, :]).long()
    return int(model(ids)[0, -1, :].argmax())


@torch.no_grad()
def causal_flip_test(model, X_te, M, A, n_examples=600, seed=0):
    """The gold-standard causal probe, independent of attention legibility.
    For each example: (1) flip the addressed cell m_a and check the
    prediction follows the new value; (2) flip a random non-addressed cell
    and check the prediction does NOT change. If the model computes exactly
    m_a, the first rate is ~1 and the second ~0, no matter what the
    attention maps look like."""
    rng = np.random.default_rng(seed)
    T = M + A
    tracks_ma = 0
    changed_by_other = 0
    n = 0
    for ids in X_te[:n_examples]:
        ctx = ids[:T].copy()
        a = address_value(ctx, M, A)
        flipped = ctx.copy()
        flipped[a] ^= 1
        if predict_bit(model, flipped) == int(flipped[a]):
            tracks_ma += 1
        base = predict_bit(model, ctx)
        others = [j for j in range(M) if j != a]
        j = int(rng.choice(others))
        other = ctx.copy()
        other[j] ^= 1
        if predict_bit(model, other) != base:
            changed_by_other += 1
        n += 1
    return tracks_ma / n, changed_by_other / n


@torch.no_grad()
def per_layer_last_row(model, ctx_np):
    """Run one example, return list over layers of the last-position
    attention row averaged over heads: each is shape (T,)."""
    ids = torch.from_numpy(ctx_np[None, :]).long()
    model(ids)
    rows = []
    for attn in model.layer_attn():            # (H, T, T)
        last = attn[:, -1, :].mean(dim=0)      # average over heads -> (T,)
        rows.append(last.cpu().numpy())
    return rows


@torch.no_grad()
def per_layer_head_last_row(model, ctx_np):
    """Return list over layers of (H, T) last-position attention per head."""
    ids = torch.from_numpy(ctx_np[None, :]).long()
    model(ids)
    return [attn[:, -1, :].cpu().numpy() for attn in model.layer_attn()]


def analyze(model, X_te, M, A, n_examples=400):
    T = M + A
    n_layers = len(model.blocks)
    # Aggregate, per layer: weight on address positions, on memory
    # positions, and on the correct cell m_a (grouped by a).
    addr_pos = list(range(M, M + A))
    mem_pos = list(range(M))
    on_addr = np.zeros(n_layers)
    on_mem = np.zeros(n_layers)
    on_correct = np.zeros(n_layers)
    count = 0
    # Per (layer, head): average weight the last-position query puts on
    # the correct memory cell m_a. Identifies the lookup head.
    H = model.blocks[0].attn.n_heads
    head_on_correct = np.zeros((n_layers, H))

    for ids in X_te[:n_examples]:
        ctx = ids[:T]
        a = address_value(ctx, M, A)
        rows = per_layer_last_row(model, ctx)
        head_rows = per_layer_head_last_row(model, ctx)
        for li in range(n_layers):
            on_addr[li] += rows[li][addr_pos].sum()
            on_mem[li] += rows[li][mem_pos].sum()
            on_correct[li] += rows[li][a]
            for h in range(H):
                head_on_correct[li, h] += head_rows[li][h, a]
        count += 1

    on_addr /= count
    on_mem /= count
    on_correct /= count
    head_on_correct /= count
    return {
        "on_addr": on_addr, "on_mem": on_mem, "on_correct": on_correct,
        "head_on_correct": head_on_correct, "n_layers": n_layers, "H": H,
    }


def _position_class(p, a, M, A):
    """Which class a position belongs to, relative to the addressed cell a.
    Returns one of: 'addressed', 'other_mem', 'addr_bits', 'readout'."""
    T = M + A
    if p == T - 1:
        return "readout"          # also the address LSB, and the readout site
    if M <= p < T - 1:
        return "addr_bits"        # address bits excluding the readout position
    if p == a:
        return "addressed"        # the cell the pointer names
    return "other_mem"            # any other memory cell


@torch.no_grad()
def causal_trace(model, X_te, M, A, n_examples=100, seed=0):
    """Causal tracing of where m_a's value flows. For each example, build a
    minimal pair differing only in the addressed cell (clean vs m_a-flipped),
    then patch the clean residual into the corrupted run at every
    (layer, position) and measure how much it recovers the clean answer.

    recovery = (logit_diff_patched - logit_diff_corrupt)
               / (logit_diff_clean - logit_diff_corrupt)
    where logit_diff is (logit[clean answer] - logit[other]) at the readout.
    recovery ~ 1 means that (layer, position) carries the m_a information;
    recovery ~ 0 means it does not. Averaged over examples within each
    position class. This is independent of attention legibility."""
    rng = np.random.default_rng(seed)
    T = M + A
    n_layers = len(model.blocks)
    classes = ["addressed", "other_mem", "addr_bits", "readout"]
    # sums[(layer, class)] and counts for averaging recovery
    sums = {(L, c): 0.0 for L in range(n_layers + 1) for c in classes}
    counts = {(L, c): 0 for L in range(n_layers + 1) for c in classes}

    used = 0
    for ids in X_te:
        if used >= n_examples:
            break
        ctx = ids[:T].copy()
        a = address_value(ctx, M, A)
        ans = int(ctx[a])
        clean = torch.from_numpy(ctx[None, :]).long()
        corrupt_np = ctx.copy()
        corrupt_np[a] ^= 1
        corrupt = torch.from_numpy(corrupt_np[None, :]).long()

        logits_clean, res_clean = model.forward_cached(clean)
        logits_corrupt, _ = model.forward_cached(corrupt)

        def ldiff(logits):
            l = logits[0, -1]
            return float(l[ans] - l[1 - ans])

        d_clean = ldiff(logits_clean)
        d_corrupt = ldiff(logits_corrupt)
        denom = d_clean - d_corrupt
        # keep only clean minimal pairs where the flip actually moved the
        # decision the expected way (denom clearly positive)
        if denom < 1.0:
            continue
        used += 1

        for L in range(n_layers + 1):
            for p in range(T):
                vec = res_clean[L][0, p]
                patched = model.forward_patched(corrupt, L, p, vec)
                rec = (ldiff(patched) - d_corrupt) / denom
                c = _position_class(p, a, M, A)
                sums[(L, c)] += rec
                counts[(L, c)] += 1

    recovery = {k: (sums[k] / counts[k] if counts[k] else float("nan"))
                for k in sums}
    return recovery, classes, n_layers, used


def plot_causal_trace(recovery, classes, n_layers, out_path):
    xs = list(range(n_layers + 1))
    labels = ["embed"] + [f"after L{L}" for L in range(1, n_layers + 1)]
    nice = {"addressed": "addressed cell $m_a$", "other_mem": "other memory",
            "addr_bits": "address bits (32-35)", "readout": "readout pos (36)"}
    markers = {"addressed": "o-", "other_mem": "s--",
               "addr_bits": "^-", "readout": "D-"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in classes:
        ys = [recovery[(L, c)] for L in xs]
        ax.plot(xs, ys, markers[c], label=nice[c])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Residual stream point (patched)")
    ax.set_ylabel("Recovery of clean answer (causal)")
    ax.set_title("Causal trace of m_a: patching clean residual into the "
                 "m_a-flipped run (M=32)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def find_lookup_head(stats):
    """The (layer, head) putting the most last-position weight on the
    addressed cell m_a. This is the dereference head."""
    hoc = stats["head_on_correct"]            # (n_layers, H)
    li, hi = np.unravel_index(int(np.argmax(hoc)), hoc.shape)
    return int(li), int(hi), float(hoc[li, hi])


@torch.no_grad()
def lookup_head_profile(model, X_te, M, A, layer_idx, head_idx,
                        n_examples=800):
    """For one head, average its last-position attention row grouped by
    address a. Returns {a -> weight that head puts on the correct cell m_a}."""
    T = M + A
    by_addr = {}
    for ids in X_te[:n_examples]:
        ctx = ids[:T]
        a = address_value(ctx, M, A)
        rows = per_layer_head_last_row(model, ctx)   # list of (H, T)
        by_addr.setdefault(a, []).append(rows[layer_idx][head_idx])
    on_correct = {a: float(np.mean(np.stack(v), axis=0)[a])
                  for a, v in by_addr.items()}
    return on_correct


def plot_lookup_head_by_address(on_correct, M, A, layer_idx, head_idx,
                                out_path):
    addrs = sorted(on_correct.keys())
    weights = [on_correct[a] for a in addrs]
    uniform = 1.0 / M
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(addrs, weights, color="#1f77b4")
    ax.axhline(uniform, color="#d62728", linestyle="--",
               label=f"uniform over {M} cells = {uniform:.3f}")
    ax.set_xlabel("Address $a$")
    ax.set_ylabel(f"L{layer_idx+1}H{head_idx} weight on cell $m_a$")
    ax.set_title(f"The lookup head (layer {layer_idx+1}, head {head_idx}) "
                 f"tracks the address: weight on $m_a$ vs $a$ (M={M})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def layer3_ensemble_on_correct(model, X_te, M, A, n_examples=800):
    """Per address, sum over layer-3 heads of the weight each head puts on
    the addressed cell m_a. Multi-head attention concatenates head outputs
    before the output projection, so the ensemble can read m_a even when no
    single head crosses a hard-spike threshold. Returns {a -> ensemble sum}."""
    T = M + A
    L3 = len(model.blocks) - 1
    by_addr = {}
    for ids in X_te[:n_examples]:
        ctx = ids[:T]
        a = address_value(ctx, M, A)
        rows = per_layer_head_last_row(model, ctx)   # list of (H, T)
        ens = float(rows[L3][:, a].sum())            # sum over heads
        by_addr.setdefault(a, []).append(ens)
    return {a: float(np.mean(v)) for a, v in by_addr.items()}


@torch.no_grad()
def head_coverage(model, X_te, M, A, layers, thresh=0.5, n_examples=800):
    """For each (layer, head) in `layers`, find the set of addresses the
    head dereferences cleanly (avg weight on m_a > thresh). Returns a dict
    (layer, head) -> set(addresses), and the union covered across all."""
    T = M + A
    acc = {}  # (li, hi) -> {a: [weights]}
    H = model.blocks[0].attn.n_heads
    for ids in X_te[:n_examples]:
        ctx = ids[:T]
        a = address_value(ctx, M, A)
        rows = per_layer_head_last_row(model, ctx)
        for li in layers:
            for hi in range(H):
                acc.setdefault((li, hi), {}).setdefault(a, []).append(
                    rows[li][hi][a])
    covered = {}
    for key, by_a in acc.items():
        cov = {a for a, ws in by_a.items()
               if np.mean(ws) > thresh}
        covered[key] = cov
    union = set()
    for cov in covered.values():
        union |= cov
    return covered, union, H


@torch.no_grad()
def ablate_layer_uniform(model, X_te, M, A, layer_idx, n_examples=400):
    """Replace one layer's attention with uniform-causal and measure acc.
    Done by temporarily monkeypatching that block's MHA forward."""
    T = M + A
    blk = model.blocks[layer_idx]
    orig_forward = blk.attn.forward

    def uniform_forward(x):
        B, Tn, D = x.shape
        H, Dh = blk.attn.n_heads, blk.attn.head_dim
        v = blk.attn.W_v(x).view(B, Tn, H, Dh).transpose(1, 2)
        causal = torch.tril(torch.ones(Tn, Tn, device=x.device))
        u = causal / causal.sum(-1, keepdim=True)
        out = (u[None, None] @ v).transpose(1, 2).contiguous().view(B, Tn, D)
        return blk.attn.W_o(out)

    blk.attn.forward = uniform_forward
    try:
        acc = evaluate(model, X_te[:n_examples], None)
    finally:
        blk.attn.forward = orig_forward
    return acc


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_progressive(stats, out_path):
    n_layers = stats["n_layers"]
    xs = np.arange(1, n_layers + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, stats["on_addr"], "o-", label="on address positions")
    ax.plot(xs, stats["on_mem"], "s-", label="on memory positions")
    ax.plot(xs, stats["on_correct"], "^-", label="on the addressed cell m_a")
    ax.set_xticks(xs)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg last-position attention weight")
    ax.set_title("Where the last-position query attends, by layer (M=32)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_layer_heatmaps(model, X_te, M, A, out_path, addrs=(0, 10, 21, 31)):
    T = M + A
    n_layers = len(model.blocks)
    picks = {}
    for ids in X_te:
        a = address_value(ids[:T], M, A)
        if a in addrs and a not in picks:
            picks[a] = ids[:T].copy()
        if len(picks) == len(addrs):
            break
    fig, axes = plt.subplots(len(picks), n_layers,
                             figsize=(4 * n_layers, 3.4 * len(picks)))
    for r, a in enumerate(sorted(picks)):
        rows = per_layer_last_row(model, picks[a])
        for li in range(n_layers):
            ax = axes[r, li]
            colors = ["#888"] * T
            if a < M:
                colors[a] = "#d62728"
            ax.bar(range(T), rows[li], color=colors)
            ax.set_ylim(0, 1.0)
            if r == 0:
                ax.set_title(f"Layer {li + 1}")
            if li == 0:
                ax.set_ylabel(f"a={a}\n(red=m{a})", fontsize=9)
            ax.set_xticks([0, M, M + A - 1])
            ax.set_xticklabels(["m0", f"m{M}/a0", f"a{A-1}"], fontsize=7)
    fig.suptitle("Last-position attention per layer, selected addresses (M=32)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    ensure_fig_dir()
    M = args.M
    A = max(3, (M - 1).bit_length())
    T = M + A
    print(f"=== Deep pointer interpretability (M={M}, A={A}, T={T}) ===")
    print(f"layers={args.layers}, heads={args.heads}, d_model={args.d_model}")

    X_tr, Y_tr = make_variant1(20000, M=M, A=A, seed=args.seed)
    X_te, Y_te = make_variant1(2000, M=M, A=A, seed=args.seed + 1)

    torch.manual_seed(args.seed)
    model = IntrospectableTransformer(
        d_model=args.d_model, n_heads=args.heads, n_layers=args.layers,
        d_ff=2 * args.d_model, max_T=T)
    print(f"parameters: {count_params(model):,}")

    if os.path.exists(CKPT) and not args.retrain:
        print(f"loading checkpoint {CKPT}")
        model.load_state_dict(torch.load(CKPT, weights_only=True))
    else:
        print(f"training {args.iters} iters ...")
        train(model, X_tr, n_iters=args.iters, seed=args.seed)
        torch.save(model.state_dict(), CKPT)
        print(f"saved checkpoint {CKPT}")

    acc = evaluate(model, X_te, Y_te)
    print(f"\nTest accuracy: {acc:.4f}")

    print("\n=== Per-layer last-position attention (averaged over test set) ===")
    stats = analyze(model, X_te, M, A, n_examples=400)
    for li in range(stats["n_layers"]):
        print(f"  layer {li + 1}: on address = {stats['on_addr'][li]:.3f}, "
              f"on memory = {stats['on_mem'][li]:.3f}, "
              f"on addressed cell m_a = {stats['on_correct'][li]:.3f}")

    print("\n=== Per-(layer, head) weight on the addressed cell m_a ===")
    for li in range(stats["n_layers"]):
        per_head = "  ".join(
            f"h{h}={stats['head_on_correct'][li, h]:.3f}"
            for h in range(stats["H"]))
        print(f"  layer {li + 1}: {per_head}")

    print("\n=== Per-layer ablation (attention -> uniform) ===")
    base = evaluate(model, X_te[:400], None)
    print(f"  baseline (first 400):     acc {base:.4f}")
    for li in range(args.layers):
        a = ablate_layer_uniform(model, X_te, M, A, li, n_examples=400)
        print(f"  layer {li + 1} attn -> uniform:  acc {a:.4f}")

    li, hi, w = find_lookup_head(stats)
    print(f"\n=== Lookup head = layer {li + 1}, head {hi} "
          f"(avg weight on m_a = {w:.3f}, uniform = {1.0 / M:.3f}, "
          f"enrichment = {w * M:.1f}x) ===")
    on_correct = lookup_head_profile(model, X_te, M, A, li, hi)
    vals = list(on_correct.values())
    print(f"  weight on m_a across {len(vals)} addresses: "
          f"min {min(vals):.3f}, mean {sum(vals)/len(vals):.3f}, "
          f"max {max(vals):.3f}")

    print("\n=== Head coverage: addresses each layer-2/3 head dereferences "
          "cleanly (weight on m_a > 0.5) ===")
    covered, union, H = head_coverage(model, X_te, M, A, layers=(1, 2))
    for li in (1, 2):
        for hi in range(H):
            cov = sorted(covered[(li, hi)])
            print(f"  layer {li + 1}, head {hi}: {len(cov):2d} addresses "
                  f"{cov if len(cov) <= 12 else cov[:12] + ['...']}")
    print(f"  union across these heads: {len(union)} of {2 ** A} addresses "
          f"covered cleanly (single head > 0.5)")

    print("\n=== Layer-3 head ensemble: total weight on m_a, per address ===")
    ens = layer3_ensemble_on_correct(model, X_te, M, A)
    vals = sorted(ens.values())
    for thr in (0.3, 0.5, 0.7):
        n = sum(1 for v in ens.values() if v > thr)
        print(f"  addresses with ensemble weight on m_a > {thr}: "
              f"{n} of {2 ** A}")
    print(f"  ensemble weight on m_a: min {vals[0]:.3f}, "
          f"mean {sum(vals)/len(vals):.3f}, max {vals[-1]:.3f}")

    print("\n=== Causal flip test (attention-independent) ===")
    tracks, other = causal_flip_test(model, X_te, M, A)
    print(f"  flip m_a -> prediction tracks the new value: {tracks:.3f}")
    print(f"  flip a random other cell -> prediction changes: {other:.3f}")

    print("\n=== Causal trace: where does m_a's value flow? "
          "(recovery by layer x position-class) ===")
    recovery, classes, nL, used = causal_trace(model, X_te, M, A,
                                                n_examples=100)
    print(f"  (averaged over {used} minimal pairs)")
    header = "  layer    " + "".join(f"{c:>16}" for c in classes)
    print(header)
    rowlabels = ["embed"] + [f"after L{L}" for L in range(1, nL + 1)]
    for L in range(nL + 1):
        row = "".join(f"{recovery[(L, c)]:>16.3f}" for c in classes)
        print(f"  {rowlabels[L]:<8}{row}")
    plot_causal_trace(recovery, classes, nL,
                      os.path.join(FIG_DIR, "causal_trace.png"))

    plot_progressive(stats, os.path.join(FIG_DIR, "progressive_attention.png"))
    plot_layer_heatmaps(model, X_te, M, A,
                        os.path.join(FIG_DIR, "layer_heatmaps.png"))
    plot_lookup_head_by_address(
        on_correct, M, A, li, hi,
        os.path.join(FIG_DIR, "lookup_head_by_address.png"))
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
