"""Decoder-only character-level Transformer on Austen + Poe + Carroll.

The architecture-axis capstone of the inductive-bias series. NumPy
implementation with every backward pass derived by hand: token embedding,
sinusoidal positional encoding, scaled dot-product attention, multi-head
attention with a causal mask, layer norm, residuals, GELU feedforward,
final logits projection, softmax cross-entropy. No autograd. The math
that scratchnn derived by hand for the MLP, CNN, and RNN posts carries
over directly; this file is what those derivations look like at
attention-scale, vectorized in NumPy because pure-Python loops over T*T
positions are not tractable.

Configuration (defaults in `main`):
  - Decoder-only (GPT-style), 2 transformer blocks
  - d_model = 64, n_heads = 4, head_dim = 16
  - FFN inner dim = 4 * d_model = 256
  - Context length T = 128
  - Char-level vocabulary plus 3 per-author style tokens
  - Adam optimizer at lr 3e-4

Style tokens. The training corpus is built by prefixing each author's
text with `[AUSTEN]`, `[POE]`, or `[CARROLL]`. At sample time, seeding
with one of those tokens steers the model into that author's style.

Run: python examples/transformer.py
"""
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

STYLE_TOKENS = ["[AUSTEN]", "[POE]", "[CARROLL]"]
AUTHOR_FILES = {
    "[AUSTEN]": "austen_pride.txt",
    "[POE]": "poe_tales.txt",
    "[CARROLL]": "carroll_looking_glass.txt",
}


# ============================================================
# Corpus
# ============================================================

def strip_gutenberg(text):
    s = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    e = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if s != -1 and e != -1:
        s = text.find("\n", s) + 1
        text = text[s:e]
    return text.strip()


def clean_whitespace(text):
    """Collapse Project Gutenberg typesetting artifacts that otherwise
    dominate a small char-level model: long runs of spaces become one
    space; sequences of 3+ newlines collapse to 2."""
    import re
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_corpus(max_chars_per_author=120_000, skip_front_matter=3000):
    """Concatenate the three corpora, each prefixed with its style token,
    after stripping the Project Gutenberg wrapper, collapsing typesetting
    whitespace, and skipping the front matter (title pages, table of
    contents) that would otherwise dominate a small char-level model's
    output distribution."""
    pieces = []
    for tok in STYLE_TOKENS:
        with open(os.path.join(DATA, AUTHOR_FILES[tok]), encoding="utf-8") as f:
            text = clean_whitespace(strip_gutenberg(f.read()))
        text = text[skip_front_matter:skip_front_matter + max_chars_per_author]
        pieces.append(tok + "\n" + text)
    return "\n".join(pieces)


def build_vocab(text):
    """Char-level vocab plus the three multi-char style tokens.

    Individual chars that also appear inside style tokens (e.g. `[`,
    `A`) stay in the vocab because they can occur outside style tokens
    too. The tokenizer matches style tokens first, then falls back to
    single chars.
    """
    chars = sorted(set(text))
    vocab = chars + STYLE_TOKENS
    s2i = {s: i for i, s in enumerate(vocab)}
    i2s = {i: s for s, i in s2i.items()}
    return vocab, s2i, i2s


def tokenize(text, s2i):
    """Tokenize: match style tokens first, fall back to chars."""
    out = []
    i = 0
    while i < len(text):
        for tok in STYLE_TOKENS:
            if text.startswith(tok, i):
                out.append(s2i[tok])
                i += len(tok)
                break
        else:
            out.append(s2i[text[i]])
            i += 1
    return np.array(out, dtype=np.int64)


# ============================================================
# Layers (each with hand-derived backward)
# ============================================================

class Embedding:
    """Token-id -> vector lookup."""

    def __init__(self, vocab_size, d_model, rng):
        self.W = rng.standard_normal((vocab_size, d_model)).astype(np.float32) * 0.02
        self.dW = np.zeros_like(self.W)
        self._ids = None

    def forward(self, ids):
        self._ids = ids
        return self.W[ids]

    def backward(self, grad):
        np.add.at(self.dW, self._ids, grad)
        return None

    def params(self):
        return [(self.W, self.dW)]


class Linear:
    """Affine map y = x W + b."""

    def __init__(self, d_in, d_out, rng):
        scale = math.sqrt(1.0 / d_in)
        self.W = rng.standard_normal((d_in, d_out)).astype(np.float32) * scale
        self.b = np.zeros(d_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad):
        x = self._x
        x_flat = x.reshape(-1, x.shape[-1])
        g_flat = grad.reshape(-1, grad.shape[-1])
        self.dW += x_flat.T @ g_flat
        self.db += g_flat.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class LayerNorm:
    """Normalize over the last dim, then scale and shift.

    For x of shape (..., D), output is gamma * (x - mean)/std + beta.
    Backward derives the input gradient through both the affine and the
    normalization (the standard Ba 2016 expression).
    """

    EPS = 1e-5

    def __init__(self, d):
        self.gamma = np.ones(d, dtype=np.float32)
        self.beta = np.zeros(d, dtype=np.float32)
        self.dgamma = np.zeros(d, dtype=np.float32)
        self.dbeta = np.zeros(d, dtype=np.float32)
        self._normed = None
        self._inv_std = None

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.EPS)
        normed = (x - mean) * inv_std
        self._normed = normed
        self._inv_std = inv_std
        return self.gamma * normed + self.beta

    def backward(self, grad):
        normed = self._normed
        inv_std = self._inv_std
        axes = tuple(range(grad.ndim - 1))
        self.dgamma += (grad * normed).sum(axis=axes)
        self.dbeta += grad.sum(axis=axes)
        d_normed = grad * self.gamma
        mean_dn = d_normed.mean(axis=-1, keepdims=True)
        mean_dn_normed = (d_normed * normed).mean(axis=-1, keepdims=True)
        return (d_normed - mean_dn - normed * mean_dn_normed) * inv_std

    def params(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    """Tanh approximation of GELU(x) = x * Phi(x)."""
    s = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(s * (x + 0.044715 * x * x * x)))


def gelu_grad(x):
    """Derivative of the tanh-approximated GELU."""
    s = math.sqrt(2.0 / math.pi)
    inner = s * (x + 0.044715 * x * x * x)
    t = np.tanh(inner)
    d_inner = s * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner


class CausalMultiHeadAttention:
    """Multi-head self-attention with a causal (lower-triangular) mask.

    Each head sees a head_dim-sized slice of the d_model embeddings.
    Forward computes Q, K, V projections, splits into heads, runs scaled
    dot-product attention per head with the future masked out, then
    concatenates and projects back to d_model. Backward derives the
    gradient through every step (softmax, scale, masking, the three
    input projections, and the output projection) by hand.
    """

    def __init__(self, d_model, n_heads, rng):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_q = Linear(d_model, d_model, rng)
        self.W_k = Linear(d_model, d_model, rng)
        self.W_v = Linear(d_model, d_model, rng)
        self.W_o = Linear(d_model, d_model, rng)

    def forward(self, x):
        T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        scale = 1.0 / math.sqrt(Dh)

        q = self.W_q.forward(x)
        k = self.W_k.forward(x)
        v = self.W_v.forward(x)
        # Split into heads: (T, D) -> (H, T, Dh)
        Q = q.reshape(T, H, Dh).transpose(1, 0, 2)
        K = k.reshape(T, H, Dh).transpose(1, 0, 2)
        V = v.reshape(T, H, Dh).transpose(1, 0, 2)

        scores = (Q @ K.transpose(0, 2, 1)) * scale          # (H, T, T)
        mask = np.tril(np.ones((T, T), dtype=np.float32))
        scores = np.where(mask[None] > 0, scores, -1e9)
        attn = softmax(scores, axis=-1)                      # (H, T, T)
        heads_out = attn @ V                                 # (H, T, Dh)
        merged = heads_out.transpose(1, 0, 2).reshape(T, D)  # (T, D)

        self._Q, self._K, self._V = Q, K, V
        self._attn = attn
        self._scale = scale
        return self.W_o.forward(merged)

    def backward(self, grad):
        T = grad.shape[0]
        H, Dh = self.n_heads, self.head_dim
        D = self.d_model

        d_merged = self.W_o.backward(grad)                   # (T, D)
        d_heads_out = d_merged.reshape(T, H, Dh).transpose(1, 0, 2)  # (H,T,Dh)

        # heads_out = attn @ V
        d_attn = d_heads_out @ self._V.transpose(0, 2, 1)    # (H, T, T)
        d_V = self._attn.transpose(0, 2, 1) @ d_heads_out    # (H, T, Dh)

        # Softmax backward, per row.
        sum_dn = (d_attn * self._attn).sum(axis=-1, keepdims=True)
        d_scores = (d_attn - sum_dn) * self._attn            # (H, T, T)

        # Scale and the unmasked entries; masked positions have attn=0 so
        # d_scores is already 0 there.
        d_scores = d_scores * self._scale

        # scores = Q @ K^T
        d_Q = d_scores @ self._K                             # (H, T, Dh)
        d_K = d_scores.transpose(0, 2, 1) @ self._Q          # (H, T, Dh)

        # Merge heads back: (H, T, Dh) -> (T, D)
        d_q = d_Q.transpose(1, 0, 2).reshape(T, D)
        d_k = d_K.transpose(1, 0, 2).reshape(T, D)
        d_v = d_V.transpose(1, 0, 2).reshape(T, D)

        # Q, K, V all read x. Sum the three input gradients.
        return (self.W_q.backward(d_q)
                + self.W_k.backward(d_k)
                + self.W_v.backward(d_v))

    def params(self):
        return (self.W_q.params() + self.W_k.params()
                + self.W_v.params() + self.W_o.params())


class FFN:
    """Position-wise feedforward: Linear -> GELU -> Linear."""

    def __init__(self, d_model, d_ff, rng):
        self.fc1 = Linear(d_model, d_ff, rng)
        self.fc2 = Linear(d_ff, d_model, rng)
        self._pre_gelu = None

    def forward(self, x):
        h = self.fc1.forward(x)
        self._pre_gelu = h
        return self.fc2.forward(gelu(h))

    def backward(self, grad):
        d_after = self.fc2.backward(grad)
        d_pre = d_after * gelu_grad(self._pre_gelu)
        return self.fc1.backward(d_pre)

    def params(self):
        return self.fc1.params() + self.fc2.params()


class TransformerBlock:
    """One GPT-style block: pre-norm + attn + residual, pre-norm + ffn + residual."""

    def __init__(self, d_model, n_heads, d_ff, rng):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, rng)

    def forward(self, x):
        a = self.attn.forward(self.ln1.forward(x))
        x = x + a
        f = self.ffn.forward(self.ln2.forward(x))
        return x + f

    def backward(self, grad):
        # grad = d/d(x + f). Residual: grad flows through both the skip
        # and the f branch.
        d_f = grad
        d_skip2 = grad
        d_ln2 = self.ffn.backward(d_f)
        d_after_attn = d_skip2 + self.ln2.backward(d_ln2)
        d_a = d_after_attn
        d_skip1 = d_after_attn
        d_ln1 = self.attn.backward(d_a)
        return d_skip1 + self.ln1.backward(d_ln1)

    def params(self):
        return (self.ln1.params() + self.attn.params()
                + self.ln2.params() + self.ffn.params())


def sinusoidal_positions(T, d_model):
    """Sinusoidal positional encoding (Vaswani 2017). No learned parameters."""
    pe = np.zeros((T, d_model), dtype=np.float32)
    pos = np.arange(T)[:, None].astype(np.float32)
    i = np.arange(d_model)[None, :].astype(np.float32)
    angle_rate = np.exp(-math.log(10000.0) * (2 * (i // 2)) / d_model)
    angles = pos * angle_rate
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


class Transformer:
    """Decoder-only character-level transformer."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256,
                 n_layers=2, max_T=128, seed=0):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.max_T = max_T
        self.pe = sinusoidal_positions(max_T, d_model)
        self.embed = Embedding(vocab_size, d_model, rng)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, rng)
                       for _ in range(n_layers)]
        self.ln_final = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng)

    def forward(self, ids):
        """ids: (T,) int. Returns logits: (T, vocab_size)."""
        T = ids.shape[0]
        x = self.embed.forward(ids) + self.pe[:T]
        for blk in self.blocks:
            x = blk.forward(x)
        return self.head.forward(self.ln_final.forward(x))

    def backward(self, grad):
        """grad: (T, vocab_size). Returns nothing; gradients accumulate."""
        g = self.head.backward(grad)
        g = self.ln_final.backward(g)
        for blk in reversed(self.blocks):
            g = blk.backward(g)
        self.embed.backward(g)  # positional encoding has no params

    def params(self):
        out = self.embed.params() + self.ln_final.params() + self.head.params()
        for blk in self.blocks:
            out += blk.params()
        return out

    def zero_grad(self):
        for _, g in self.params():
            g.fill(0.0)


# ============================================================
# Loss
# ============================================================

def softmax_cross_entropy(logits, targets):
    """logits: (T, V), targets: (T,) int. Returns (mean loss, grad w.r.t. logits)."""
    T, V = logits.shape
    log_z = np.log(np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(
        axis=-1, keepdims=True)) + logits.max(axis=-1, keepdims=True)
    log_probs = logits - log_z
    loss = -log_probs[np.arange(T), targets].mean()
    probs = np.exp(log_probs)
    grad = probs.copy()
    grad[np.arange(T), targets] -= 1.0
    grad /= T  # mean over T
    return loss, grad


# ============================================================
# Adam
# ============================================================

class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.params = params

    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# Training
# ============================================================

def get_batch(ids, T, batch_size, rng):
    """Sample `batch_size` random (input, target) windows of length T."""
    starts = rng.integers(0, len(ids) - T - 1, size=batch_size)
    x = np.stack([ids[s:s + T] for s in starts])
    y = np.stack([ids[s + 1:s + T + 1] for s in starts])
    return x, y


def train_step(model, x_batch, y_batch, optimizer):
    """One Adam step over a batch of sequences (processed one at a time)."""
    model.zero_grad()
    total = 0.0
    B = x_batch.shape[0]
    for b in range(B):
        logits = model.forward(x_batch[b])
        loss, dlogits = softmax_cross_entropy(logits, y_batch[b])
        total += loss
        # Scale grad by 1/B so we accumulate the mean of B examples.
        model.backward(dlogits / B)
    optimizer.step()
    return total / B


def sample_text(model, s2i, i2s, seed_text, n_steps, T, temperature=0.9):
    """Greedy/temperature sampling. Tokenize seed, then extend n_steps tokens."""
    ids = tokenize(seed_text, s2i).tolist()
    for _ in range(n_steps):
        ctx = ids[-T:]
        ctx_arr = np.array(ctx, dtype=np.int64)
        logits = model.forward(ctx_arr)
        last_logits = logits[-1] / temperature
        probs = softmax(last_logits)
        next_id = int(np.random.choice(len(probs), p=probs / probs.sum()))
        ids.append(next_id)
    return "".join(i2s[i] for i in ids)


def main(seed=0, T=128, d_model=64, n_heads=4, d_ff=256, n_layers=2,
         max_chars_per_author=120_000, batch_size=32, lr=3e-4,
         n_iters=2000, log_every=100, sample_every=500, sample_n=220):
    print("Loading corpus...")
    text = load_corpus(max_chars_per_author=max_chars_per_author)
    vocab, s2i, i2s = build_vocab(text)
    ids = tokenize(text, s2i)
    print(f"  corpus: {len(text)} chars -> {len(ids)} tokens; vocab: {len(vocab)}")

    print("Building model...")
    model = Transformer(len(vocab), d_model=d_model, n_heads=n_heads,
                        d_ff=d_ff, n_layers=n_layers, max_T=T, seed=seed)
    n_params = sum(p.size for p, _ in model.params())
    print(f"  parameters: {n_params:,}")
    optimizer = Adam(model.params(), lr=lr)

    rng = np.random.default_rng(seed + 1)
    np.random.seed(seed + 2)  # for sampling
    start = time.time()
    running = 0.0
    history = []
    for it in range(1, n_iters + 1):
        x_b, y_b = get_batch(ids, T, batch_size, rng)
        loss = train_step(model, x_b, y_b, optimizer)
        running += loss
        if it % log_every == 0:
            avg = running / log_every
            history.append(avg)
            elapsed = time.time() - start
            print(f"  iter {it:5d}  loss {avg:.4f}  ({elapsed:.0f}s elapsed)")
            running = 0.0
        if sample_every and it % sample_every == 0:
            for tok in STYLE_TOKENS:
                s = sample_text(model, s2i, i2s, tok + "\n", sample_n, T)
                print(f"\n--- sample seeded by {tok} ---")
                print(s)
            print()

    return model, s2i, i2s, history


if __name__ == "__main__":
    main()
