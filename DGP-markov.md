**“A Tiny Transformer Learns a Tiny Language”**

The point is not to make a useful model. The point is to make the transformer’s inductive bias visible.

You want a DGP simple enough that the reader can fully understand the rule, but nontrivial enough that logistic regression / MLP / RNN / CNN comparisons are meaningful.

## Good DGP choices

I would avoid making it purely “last 4 bits → next bit” at first, because a shallow MLP over the last 4 bits can solve that cleanly. That is useful as a baseline, but it does not really justify attention.

Better: use a family of rules where the relevant context position varies.

### Option A: fixed-order Markov bit generator

Example:

[
x_t = x_{t-1} \oplus x_{t-3} \oplus x_{t-4}
]

This is clean and deterministic. The model needs to learn which past positions matter. But since the offsets are fixed, a convolution or MLP over a context window works very well.

Useful lesson:

> Not all sequence problems need transformers. If the dependency structure is local and fixed, CNNs or small MLPs may be more sample-efficient.

This is good as the opening example, not the final one.

---

## Better capstone DGP: content-addressed bit sequences

Make the next bit depend on **where a marker occurs**, not just a fixed offset.

For example, generate sequences over bits plus maybe a separator token:

```text
0 1 1 0 1 0 0 1  | query: 3 | answer: third previous bit
```

But to keep the alphabet binary, you can encode the task entirely in bits.

A simple version:

```text
[payload bits] [address bits] [target bit]
```

Example:

```text
payload: 1 0 1 1
address: 0 1
target: payload[1] = 0
```

So the sequence is:

```text
1 0 1 1 0 1 0
```

The model sees:

```text
1 0 1 1 0 1
```

and predicts the final bit:

```text
0
```

Here the final two bits encode an address into the previous four payload bits.

This is much more transformer-ish. The model has to use the address bits to decide **which earlier position to attend to**.

That gives you a tiny synthetic “retrieval” task.

## Why this is a better transformer teaching example

The DGP is still simple:

[
\text{payload} = (b_0,b_1,b_2,b_3)
]

[
\text{address} = (a_0,a_1)
]

[
i = 2a_0 + a_1
]

[
y = b_i
]

The model must predict (y).

But the dependency is not fixed. Sometimes the answer depends on (b_0), sometimes (b_1), sometimes (b_2), sometimes (b_3).

That makes attention natural:

> The query is determined by the address. The keys are determined by positions. The values carry the payload bits.

That is exactly the kind of story you want.

## Suggested blog post arc

### 1. Start with the real lesson

Say something like:

> A transformer is not magic. It is a differentiable mechanism for routing information between positions in a sequence. The attention mechanism decides which earlier tokens are relevant to the current prediction.

Then introduce the toy DGP.

### 2. Define the vocabulary

Binary tokens:

```python
vocab = {0: [1, 0], 1: [0, 1]}
```

But I would also include learned token embeddings, because one-hot embeddings are useful pedagogically but not how transformers are usually implemented.

Maybe do both:

```python
x_onehot = one_hot(bits)
x = x_onehot @ W_embed
```

That lets you show that an embedding layer is just a learned lookup table.

### 3. Add positional embeddings

This is crucial. Without positional information, attention cannot distinguish “the first payload bit” from “the third payload bit” if the token values are identical.

```python
h_t = token_embedding[x_t] + position_embedding[t]
```

This is a great teaching point:

> Attention sees a set of vectors. Positional embeddings tell it where each vector came from.

### 4. Single-head causal self-attention

For each position (t):

[
q_t = h_t W_Q
]

[
k_j = h_j W_K
]

[
v_j = h_j W_V
]

[
\alpha_{tj}
===========

\frac{
\exp(q_t k_j^\top / \sqrt{d_k})
}{
\sum_{m \leq t} \exp(q_t k_m^\top / \sqrt{d_k})
}
]

[
z_t = \sum_{j \leq t} \alpha_{tj} v_j
]

The important thing to show visually is the attention pattern from the final position back to the payload positions.

For this DGP, the model should learn something like:

```text
address 00 -> attend to payload position 0
address 01 -> attend to payload position 1
address 10 -> attend to payload position 2
address 11 -> attend to payload position 3
```

That makes the abstract machinery concrete.

### 5. Why one head may struggle

A single head has to do multiple things at once:

1. read the address bits,
2. map the address to a payload position,
3. retrieve the payload bit,
4. pass the retrieved bit to the output classifier.

It may learn the task, but with limited dimension, limited data, or noise, it can be awkward.

Then introduce two heads:

```text
Head 1: interpret/address the query
Head 2: retrieve/copy the relevant payload bit
```

Careful caveat: the heads may not actually specialize so neatly unless you force or regularize it. But this is still a useful conceptual decomposition.

The honest phrasing is:

> Multi-head attention gives the model multiple routing mechanisms in parallel. It does not guarantee a clean human-interpretable division of labor, but it makes such decompositions available to gradient descent.

That is a nice skeptical sentence.

## Implementation plan

I’d implement it in stages.

### Dataset generator

```python
import random

def sample_example(payload_len=4):
    payload = [random.randint(0, 1) for _ in range(payload_len)]
    index = random.randrange(payload_len)

    # for payload_len = 4
    addr = [(index >> 1) & 1, index & 1]

    x = payload + addr
    y = payload[index]
    return x, y, index
```

Then train on examples like:

```python
x = [1, 0, 1, 1, 0, 1]
y = 0
```

This is not next-token language modeling yet. It is sequence-to-label prediction.

Then once that works, turn it into autoregressive next-token prediction by appending the answer token and training the model to predict the next token at each position, with the loss focused on the final prediction.

### Minimal transformer block

For the blog, I’d avoid PyTorch `nn.TransformerEncoderLayer` at first. Write the attention manually.

Skeleton:

```python
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h):
        Q = self.W_q(h)
        K = self.W_k(h)
        V = self.W_v(h)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))

        # causal mask
        T = h.size(1)
        mask = torch.tril(torch.ones(T, T, device=h.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        A = torch.softmax(scores, dim=-1)
        Z = A @ V
        return Z, A
```

Then the full model:

```python
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=2, seq_len=6, d_model=32, d_k=16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = SingleHeadSelfAttention(d_model, d_k)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 2)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        h = self.token_emb(x) + self.pos_emb(pos)
        z, A = self.attn(h)
        h = self.ln(h + z)

        logits = self.out(h[:, -1])
        return logits, A
```

Then loss:

```python
loss = F.cross_entropy(logits, y)
```

## Experiments worth showing

The post becomes much better if you include tiny ablations.

### 1. No positional embeddings

Prediction should fail or degrade badly.

Lesson:

> Attention without position is permutation-equivariant. It can know what tokens exist, but not where they are.

### 2. Fixed-offset DGP versus address-based DGP

Compare:

```python
y = x[-1] ^ x[-3] ^ x[-4]
```

versus:

```python
y = payload[address]
```

Lesson:

> Fixed local rules are not where transformers shine. Conditional retrieval is a better toy model for attention.

### 3. Single-head versus two-head attention

Measure training steps to reach 99% accuracy.

You might get results like:

```text
MLP over full context: strong baseline
Single-head transformer: learns, but sometimes slower
Two-head transformer: more stable / faster
No position embedding: fails
RNN: learns but less parallel / sometimes harder
CNN: good on fixed-offset rule, weaker on content-addressed lookup
```

Do not overclaim. On tiny tasks, MLPs often win.

The intellectually honest point is:

> Transformers are not universally more sample-efficient. Their advantage appears when the DGP contains reusable, compositional, content-addressed dependencies that scale with sequence length and data diversity.

## Even better DGP: variable-length retrieval

Once the fixed payload length works, make the payload length variable.

Example:

```text
payload length: n
address length: ceil(log2 n)
target: payload[address]
```

Now the same learned mechanism can generalize somewhat across positions and lengths.

That gives you the transition to real language:

> In natural language, the relevant earlier token is not usually at a fixed offset. It depends on syntax, semantics, discourse state, and task context. Attention gives the model a learnable routing mechanism over the context.

## The capstone thesis

I’d make the capstone’s central claim something like:

> The transformer is an architecture for learned, differentiable information routing. In a toy bit-sequence world, attention learns to route information from the relevant past position to the current prediction. In real corpora, the same mechanism scales to routing over syntax, references, entities, code variables, long-range dependencies, and latent semantic structure.

That connects your DGP framing nicely:

| Architecture        | Inductive bias                           |
| ------------------- | ---------------------------------------- |
| Logistic regression | linear decision boundary                 |
| MLP                 | arbitrary fixed-size nonlinear function  |
| CNN                 | local translation/shared-pattern bias    |
| RNN                 | sequential state update bias             |
| Transformer         | content-addressed routing over a context |

## My recommendation

Use **two DGPs** in the post:

1. **Fixed Markov bit rule**
   Shows that simple sequence prediction does not require attention.

2. **Addressed payload lookup**
   Shows why attention is useful.

That contrast will make the transformer lesson much sharper.

The addressed lookup task is probably your best capstone toy problem. It is simple enough to explain completely, but rich enough to expose embeddings, positional encodings, queries, keys, values, attention maps, multi-head attention, ablations, and scaling intuition.
