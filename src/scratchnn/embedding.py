"""Embedding (lookup table) layer for scratchnn.

An `Embedding` maps a discrete token id (int) to a dense embedding vector
(`list[float]`). Mathematically it is a `Linear` layer applied to a
one-hot encoded input, but it skips the multiply-by-zero by indexing
directly. Real frameworks (`torch.nn.Embedding`, etc.) make the same
move for the same reason.

The cell maintains an internal LIFO cache of the token ids passed to
`forward`, so `backward` accumulates each gradient into the correct row
when the same `Embedding` is called many times per forward pass (the
Bengio 2003 fixed-context language model does exactly this, calling the
embedding once per context position).

Storage matches `Linear`'s row convention: one flat weight vector
(`list[float]` of length `embed_dim`) per vocabulary token, with the
matching gradient accumulator. `parameters()` yields one
`(values, grads)` pair per row, so `step` and `zero_grad` work
unchanged.
"""
import math
import random

from .neural_net import Layer


class Embedding(Layer):
    """Token-id to embedding-vector lookup.

    `forward(token_id)` returns a fresh copy of the embedding for
    `token_id` and records the id in the cache stack.
    `backward(grad)` pops the most-recent id and accumulates `grad`
    into that row's gradient. Returns `None` (the input was discrete,
    so there is no upstream gradient to propagate).
    """

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        r = 1.0 / math.sqrt(embed_dim)
        self.weights = [[random.uniform(-r, r) for _ in range(embed_dim)]
                        for _ in range(vocab_size)]
        self.dweights = [[0.0 for _ in range(embed_dim)]
                         for _ in range(vocab_size)]
        self.cache = []

    def reset_cache(self):
        """Clear the LIFO id cache. Call before each forward pass."""
        self.cache = []

    def forward(self, token_id):
        self.cache.append(token_id)
        return list(self.weights[token_id])

    def backward(self, grad):
        token_id = self.cache.pop()
        row = self.dweights[token_id]
        for k in range(len(grad)):
            row[k] += grad[k]
        return None

    def parameters(self):
        return list(zip(self.weights, self.dweights))


class EmbedConcat(Layer):
    """Embed `context_len` token ids and concatenate into one flat vector.

    Wraps an `Embedding`. `forward(ids)` takes a `list[int]` of length
    `context_len`, calls the embedding once per token, and concatenates
    the results into a `list[float]` of length `context_len * embed_dim`.
    `backward(g)` splits `g` into `context_len` slices and routes each
    back through the embedding in LIFO order (matching the embedding's
    cache discipline).

    Use case: the input layer of a fixed-context (Bengio 2003) language
    model. The output sits at the boundary of a normal MLP body, so the
    rest of the `Network` (`Linear`, `Tanh`, `SoftmaxCrossEntropy`)
    composes without change.

    Parameters delegate to the wrapped embedding, so the table is the
    only place the rows live.
    """

    def __init__(self, vocab_size, embed_dim, context_len):
        self.context_len = context_len
        self.embed_dim = embed_dim
        self.embed = Embedding(vocab_size, embed_dim)

    def forward(self, ids):
        self.embed.reset_cache()
        out = []
        for token_id in ids:
            out.extend(self.embed.forward(token_id))
        return out

    def backward(self, g):
        d = self.embed_dim
        # Backward in reverse to match the embedding's LIFO cache.
        for i in reversed(range(self.context_len)):
            chunk = g[i * d:(i + 1) * d]
            self.embed.backward(chunk)
        return None

    def parameters(self):
        return self.embed.parameters()
