# Transformer: Attention as a Sequence Prior (Outline)

## Title and thesis

**Working title:** "Transformers: Attention as a Sequence Prior, and What It Gives Up"

**One-line thesis:** A Transformer is permutation-equivariant over positions by construction; a positional encoding is the smallest possible inductive bias added back to recover a sequence prior, and attention is the mechanism by which any position may read from any other in one step.

**Capstone framing:** This is post 4 of 4. The series asked one question across four architectures: *what does this architecture assume about the data?* The Transformer is where the answer becomes sharpest, because the architecture starts from *almost no* assumption (a set of vectors) and adds structure back in pieces that are individually small and individually inspectable. The whole series lands here.

## Pedagogical arc (the thread the series has been pulling)

1. **scratchnn (post 1).** A `Linear` layer assumes nothing about the input but its dimensionality. Permute the input features and retrain and you recover the same fit. The MLP's only structural assumption is feature exchangeability.
2. **CNN (post 2).** Translation equivariance in space. A learned filter applied at every location: convolution is parameter sharing across a translation group. The assumption: nearby pixels are more related than distant ones, and the relation does not depend on absolute position.
3. **RNN (post 3).** Time-translation equivariance. The same cell computation at every step; the hidden state carries the past forward. The assumption: temporal locality matters, and the *function* mapping (state, input) to next state is constant in time.
4. **Transformer (post 4, here).** Permutation equivariance over positions, then deliberately broken. The model treats its input as a *set* of vectors and uses attention to let any vector look at any other. Adding a positional encoding turns the set into a sequence. The assumption is minimal and *learned*: there is no built-in notion of "nearby" the way a CNN has, only the inductive bias that two tokens at similar positions have similar position vectors.

The pattern across the series: each architecture is a different way of writing down "what stays the same under what transformation," and that symmetry choice *is* the inductive bias. Transformers do the least of this hand-wiring and let attention learn the rest. That is the whole story.

## Section-by-section breakdown

### Section 1. Setup: from sequences to sets of vectors

- A token sequence $x_1, \ldots, x_L$ becomes an embedding matrix $X \in \mathbb{R}^{L \times d}$.
- Two questions: (a) how does row $i$ aggregate information from the other rows, and (b) how does the model know that row $i$ came *before* or *after* row $j$?
- Attention answers (a). Positional encoding answers (b). Everything else is plumbing.

### Section 2. Scaled dot-product attention, derived

- Start from a simpler question: given a query vector $q$, how do we retrieve a soft-weighted average of a set of value vectors $v_1, \ldots, v_L$, keyed by $k_1, \ldots, k_L$?
- Score by similarity: $s_j = q^\top k_j$. Normalize to weights: $\alpha_j = \mathrm{softmax}(s)_j$. Aggregate: $\sum_j \alpha_j v_j$.
- Stack queries: $Q \in \mathbb{R}^{L \times d_k}$, $K \in \mathbb{R}^{L \times d_k}$, $V \in \mathbb{R}^{L \times d_v}$. The full operation,
  $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$
- This is the same softmax we already met in scratchnn, just applied row-wise to a matrix of scores.

### Section 3. Why $\sqrt{d_k}$? A variance argument.

- Suppose $q$ and $k$ have i.i.d. entries with mean $0$, variance $1$. Then
  $$\mathrm{Var}(q^\top k) = \sum_{i=1}^{d_k} \mathrm{Var}(q_i k_i) = d_k.$$
- The standard deviation of the raw score grows like $\sqrt{d_k}$. Plug a score of magnitude $\sqrt{d_k}$ into softmax and, as $d_k$ grows, the softmax saturates: one weight near $1$, the rest near $0$. The gradient of softmax in this regime is near zero (this is the same plateau we saw with the sigmoid in scratchnn).
- Dividing scores by $\sqrt{d_k}$ holds their variance at $1$ regardless of $d_k$. Softmax stays in its informative range; gradients keep flowing.
- Connect back to scratchnn: this is the same "keep the pre-softmax range bounded" pressure we managed with the max-subtraction trick. Different problem, same intuition.

### Section 4. Multi-head attention as parallel subspace attentions

- One attention head: project $X$ to $Q, K, V$ via learned matrices $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$, attend, return $L \times d_v$.
- Multi-head: $h$ independent attentions on $h$ learned projections, concatenated, then a final linear projection $W_O$ back to dimension $d$.
- Why parallel heads instead of a single bigger attention? A single softmax must commit to one mixture of value vectors per query. Heads let the model pull different relations from the same input simultaneously: one head can track local syntax, another long-range coreference, another none of the above. Each head gets a *smaller* $d_k$ (typically $d / h$), so the total parameter count is comparable to one wide head; the gain is the expressiveness of multiple coexistent mixtures.
- Inductive bias note: the heads are *not* told what to specialize on. The architecture provides the seats, training fills them. This is in the same spirit as letting a CNN's filters discover edges rather than hand-coding them.

### Section 5. Positional encoding: re-introducing sequence order

- The motivating fact: $\mathrm{Attention}(PX, PK, PV)$ for a permutation matrix $P$ permutes the output rows the same way and leaves their *content* unchanged. Attention does not see order.
- So permute the words of a sentence and the model emits the same multiset of token representations, just reshuffled. That cannot work for language.
- **Two cures**, both add a position-dependent vector to each embedding:
  - **Sinusoidal (fixed):** $\mathrm{PE}(p, 2i) = \sin(p / 10000^{2i/d})$, $\mathrm{PE}(p, 2i+1) = \cos(p / 10000^{2i/d})$. No parameters, generalizes to lengths unseen in training, and there is a nice identity: a relative offset $p \to p + \Delta$ acts as a linear map on the pair $(\sin, \cos)$, so dot-product attention can in principle pick up relative positions.
  - **Learned:** a parameter matrix $E \in \mathbb{R}^{L_{\max} \times d}$, one row per position, added to the embedding. Simpler, no extrapolation beyond $L_{\max}$.
- This is the most important sentence of the section: positional encoding is the architecture's *only* inductive bias about order. Make it weaker and the model has more freedom and less prior; make it stronger (relative-position biases, ALiBi, RoPE) and you encode more about sequences. Each is a different point on the same trade-off.

### Section 6. Causal masking for autoregressive generation

- For language modeling we want $p(x_t \mid x_{<t})$. Position $t$ must not attend to positions $> t$ during training, else the model trivially copies the next token through attention.
- Implementation: set the *raw scores* $s_{tj}$ to $-\infty$ for $j > t$ before softmax. After softmax, those weights are exactly zero; the gradient through them is exactly zero.
- This is one line of code and a small drawing in the post. The drawing matters: it makes "causal mask" obvious instead of mysterious.

### Section 7. Layer norm and residual connections

- A Transformer block is two sub-layers (attention, then a position-wise MLP), each wrapped in residual + layer-norm:
  $$y = \mathrm{LN}(x + \mathrm{Sublayer}(x)).$$
- **Residual** ($x + f(x)$): gradients pass through $1 + f'(x)$ rather than $f'(x)$ alone. With enough stacked layers, $\prod_\ell f_\ell'(x_\ell)$ can vanish or explode; the $+ x$ keeps a gradient highway open. This is the same idea ResNets used for image classification.
- **Layer norm**: normalize each position's vector to mean $0$, variance $1$, then apply a learned per-dimension scale and shift. Unlike batch norm, the statistic is computed *per position, per example*, so there is no dependence on batch size. The math: with $x \in \mathbb{R}^d$,
  $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i,$$
  where $\mu, \sigma^2$ are the mean and variance over the $d$ components of *that one* $x$.
- Why layer norm rather than batch norm here: sequences come in different lengths, batch statistics across positions would mix unrelated tokens, and inference often runs at batch size 1. Layer norm sidesteps all of that.
- The position-wise MLP: a two-layer MLP applied to each position independently (same parameters across positions). This is literally a `Linear, ReLU, Linear` block from scratchnn, broadcast over $L$ positions.

### Section 8. Putting one block together

- Input: $X \in \mathbb{R}^{L \times d}$ (token embeddings plus positional encoding).
- Multi-head self-attention, with causal mask, residual, layer norm.
- Position-wise MLP, residual, layer norm.
- Output: $X' \in \mathbb{R}^{L \times d}$, ready for the next block. Stack $N$ blocks. End with one final `Linear` to vocabulary size; softmax + cross-entropy at training time. The output activation is still in the loss; the model still emits logits. The scratchnn organizing principle survives.

### Section 9. Inductive bias table (the through-line)

| Architecture | Equivariance / invariance | What is shared | What is hand-wired | What is learned |
|---|---|---|---|---|
| MLP | feature exchangeability only | nothing across inputs | a flat input | everything |
| CNN | translation in space | one filter applied everywhere | local receptive field, spatial grid | filter weights |
| RNN | translation in time | one cell at every step | sequential structure, locality through state | cell weights |
| Transformer | permutation over positions, **broken by positional encoding** | one attention computation across all pairs | only the positional encoding | weights *and* what counts as "near" in position space |

Key reading of the table: the Transformer hand-wires the *least*. Locality, neighborhood, and even "the next word matters more than a word ten paragraphs ago" are all things the model can learn or refuse to learn. The CNN forces locality on you; the Transformer offers it as a hypothesis the data must support.

### Section 10. What the Transformer gives up

- **Computational cost.** Self-attention is $O(L^2 \cdot d)$ in time and memory. An RNN is $O(L \cdot d^2)$ in time, $O(d)$ in state. For long sequences the $L^2$ term dominates, and most modern long-context work is some scheme for breaking that quadratic (sparse, local, linearized, state-space hybrids).
- **The recurrence prior.** The RNN's "process token by token" prior is genuinely informative for some sequences, especially short ones with clean temporal structure. A Transformer trained on a tiny corpus often underperforms an RNN trained on the same data: it has more freedom than the data can constrain.
- **A built-in notion of distance.** CNNs and RNNs build "nearby tokens matter more" into the architecture; Transformers must learn it through positional encoding and attention patterns. With small data this is a real disadvantage.

### Section 11. What the Transformer gains

- **Arbitrary-distance dependencies in one layer.** Position $1$ and position $L$ are equally close in attention's graph. An RNN would need $L$ steps to thread information between them, and information decays along the way.
- **Full parallelism across positions.** Every output position is computed from $Q, K, V$ in one matrix multiply. An RNN must process step $t$ before step $t+1$. This is the reason the Transformer scaled and the LSTM did not.
- **Learned rather than hand-crafted locality.** The model can attend strictly to nearby tokens *or* skip to distant ones, per head, per layer. Locality is a *finding* in the attention maps, not an assumption.

## Worked example: character-level Transformer on two-author Public-Domain prose

**Authors (pick two, propose three for flexibility):**

- **Jane Austen** (e.g. *Pride and Prejudice*, *Emma*): measured, syntactically clean, free indirect style.
- **Edgar Allan Poe** (e.g. selected tales): ornate, paratactic, heavy on dependent clauses and Latinate diction.
- **Lewis Carroll** (e.g. *Alice in Wonderland*, *Through the Looking-Glass*): playful, dialogue-heavy, with nonsense words.

These three have stylistic differences that are *visible at the character level*, which matters because the model is char-level. (Shakespeare is also tempting but the verse line breaks are a different beast; Twain's dialect spelling is great but it muddies the basic question.)

**Corpus.** Aim for roughly $200\text{ KB}$ to $500\text{ KB}$ per author, all from Project Gutenberg. Strip the Gutenberg license header and footer, normalize whitespace, leave punctuation. Manageable to train a small Transformer on in minutes on a laptop.

**Style conditioning.** Three options the post can compare:

1. **Per-author BOS token.** Prepend `<AUSTEN>`, `<POE>`, `<CARROLL>` as the first token of every sample. The model learns to read the BOS token as a style switch through attention.
2. **Inline prefix tag.** Same idea but as a literal string like `[STYLE=POE]\n` followed by text. Slightly less clean but easier to inspect.
3. **Separate small models per author.** A useful baseline to compare to the conditioned single model. The single model with style tokens should fit the joint distribution; the separate models can only fit one each.

**Recommended path:** option 1 (BOS token), with option 3 as a sanity baseline. The capstone figure is "same model, three styles, different tokens" rather than "three models."

**Architecture for the worked example.**

- Vocabulary: ASCII printable (~100 tokens) plus three style tokens.
- Embedding dim $d = 128$, 4 heads ($d_k = 32$), 4 blocks, sequence length $L = 256$.
- Roughly $1\text{M}$ parameters. Trains in minutes; small enough to read about.

**Training.** Cross-entropy on next-character prediction with causal mask. AdamW (acknowledged as a deliberate departure from scratchnn's plain SGD; see code section). Maybe 5k to 20k steps depending on hardware.

**Generation.** Sample with temperature and top-$k$. Show:

- Same seed text, different style tokens, contrast the continuations.
- Reverse: feed a style token plus an empty prompt and let it free-run. The model should drift toward that author's habits.

## Math content: the centerpiece

All derivations should mirror the walkthrough's tone: introduce the question, write the smallest expression that answers it, then read it back in words. Specifics to derive in full, not drop in:

- **Dot-product similarity as soft retrieval.** Start from "I want a weighted average where the weights reflect how well each key matches the query." Show why $q^\top k$ is the natural choice if we are working with vectors in $\mathbb{R}^{d_k}$ and want a scalar score. Show why we *don't* normalize $k$ to unit length (an ablation cited).
- **The $\sqrt{d_k}$ scaling.** Carry out the variance calculation (section 3 above). Connect to "softmax with large inputs is essentially argmax, gradient is zero."
- **Backprop through attention.** Sketch the gradient through the three pieces: softmax (Jacobian known from scratchnn), the matmul $QK^\top$ (Linear's gradient generalized), and the matmul with $V$. Note that we do not have to hand-write any of this if we use a framework, *but* the derivation mirrors scratchnn's `Linear.backward` exactly.
- **Multi-head as block-diagonal projection.** Show that concatenating $h$ heads each with $d/h$ dimensions and projecting through $W_O$ is mathematically equivalent to one wide attention with a constrained, block-diagonal $W_Q, W_K, W_V$. The block-diagonal constraint is the inductive bias.
- **Permutation equivariance, formally.** With $P$ a permutation matrix and self-attention $\mathrm{SA}$, $\mathrm{SA}(PX) = P \cdot \mathrm{SA}(X)$. One-line proof: the softmax is row-stochastic and equivariant under permutation of both rows and columns. Conclusion: without positional encoding, the model treats the input as a set.
- **Positional encoding as a relative-position kernel.** For sinusoidal $\mathrm{PE}$, show that $\langle \mathrm{PE}(p), \mathrm{PE}(p + \Delta) \rangle$ depends only on $\Delta$, not on $p$. So dot-product attention between two positionally-encoded embeddings has a baked-in (additive) relative-position term.
- **Layer norm gradient.** Brief: derivative of $(x - \mu) / \sqrt{\sigma^2 + \epsilon}$ with respect to $x_j$ is not the obvious thing because $\mu$ and $\sigma$ depend on $x$. Give the formula or, more pedagogically, point out that this is exactly the kind of "if you build it from scalar autograd, you don't have to derive this by hand" payoff promised at the end of scratchnn's walkthrough.

## Code approach

**Honest acknowledgement, up front in the post:** scratchnn's pure-Python pedagogy ended at the MLP. A Transformer block in pure Python would be unreadable (nested list comprehensions over $L \times L \times h \times d$ tensors) *and* unrunnable (a 256-token sequence already wants $L^2 d h \approx 4\text{M}$ multiplications per attention; ten layers and a few hundred training steps would take hours). The pedagogy of "everything is a list of floats" has expired.

**What we switch to: PyTorch, minimally.**

- We do not use any layer above `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`. Specifically *no* `nn.MultiheadAttention`, *no* `nn.TransformerEncoderLayer`. We write attention from the matrix expression in section 2. The point is to see the math.
- We let autograd handle backward. This is the moment the "from per-layer to per-operation" closing note of scratchnn's walkthrough is cashed in. Each derivation we did still tells us what backward *would* be; we just no longer write it.
- The math derivations carry over directly. The only thing we are buying is vectorization (`@` instead of nested loops) and a GPU path if we want one.

NumPy is a possible alternative (closer to scratchnn's spirit, no autograd), but training without autograd would be a regression: we'd lose the gradient checker we already wrote *and* lose autograd's correctness guarantees. PyTorch with hand-written attention is the honest middle.

## Figures to produce

1. **Attention heatmaps.** A $L \times L$ matrix per head per layer, colored by attention weight. Show how heads in early layers attend locally (positional structure) while later-layer heads spread across the sequence (semantic structure). This is the figure that makes the inductive-bias story land visually.
2. **Training loss curves.** Cross-entropy per step, log scale on $y$. Useful to see learning-rate effects; a small ablation table on $d_k$ or number of heads can hang off this.
3. **Generated text samples, per style token.** The payoff figure. Three columns (Austen, Poe, Carroll), each a generated continuation of a neutral prompt. Annotate one or two style markers in each (e.g. Austen's dependent clauses, Poe's parataxis, Carroll's nonsense diction).
4. **Parameter count comparison across the series.** A table for the four architectures, each trained on a comparable task (or normalized to "params for similar accuracy on MNIST" if we want to be cute):
   - MLP: parameters scale with input size $\times$ hidden width.
   - CNN: parameters scale with kernel size $\times$ channels, *not* image size. (The first sharp drop in the table.)
   - RNN: parameters scale with hidden dim squared, independent of sequence length.
   - Transformer: parameters scale with $d^2 \cdot N_{\text{blocks}}$, *but* compute scales with $L^2 \cdot d$ per layer.
5. **Attention as soft retrieval, schematic.** A diagram: a query vector, a row of (key, value) pairs, softmax weights on the keys, the output as a weighted sum of values. Captioned: "an MLP is hard-coded function evaluation; attention is soft, learned table lookup."
6. **Permutation equivariance figure.** Two side-by-side runs: same input tokens, different order, *no positional encoding*, output multisets equal. With positional encoding, output multisets differ. One picture, the inductive-bias claim made concrete.

## Closing reflection

What the series found: every architecture is the same neural-network primitive (a stack of differentiable functions trained by backprop) wrapped in a different symmetry. The MLP's symmetry was trivial; the CNN's was spatial; the RNN's was temporal; the Transformer's was a deliberately weak symmetry over positions that the data is allowed to break through learned attention patterns. None of these architectures invented new math. They each made a different choice about *what should stay constant under what transformation*, and the rest of the design fell out of that choice.

A reader who has walked from scratchnn through the four posts should now look at any new architecture (mixers, state-space models, graph networks, etc.) and immediately ask: *what symmetry is this assuming, and what is it learning?* That habit is the take-away.

**Where the series leaves the reader:**

- *Foundation models and pretraining* are the natural next layer: a Transformer is the architecture, but the modern story is about scale of data and compute, transfer, instruction-following, alignment. None of this is in scope here; the inductive-bias lens still applies, but the questions are different (what does pretraining encode, what does fine-tuning preserve), and that is a different series.
- *Scaling laws* are an empirical companion to the same observation: with a weak enough inductive bias and enough data, the architecture stops being the bottleneck. The Transformer is the canonical example. The question of whether *something even weaker* would work better at extreme scale is open.
- The mechanical move from per-layer backward to per-operation autograd (scratchnn's closing note) is no longer a teaser. We used it here. The natural next series is "build the autograd engine that made all of this possible," which is a different exercise in the same spirit.

## Notes on tone and conformance (for the writer)

- No em-dashes. Use commas, colons, periods, or parentheses.
- Match the walkthrough's voice: short paragraphs, math integrated, no marketing.
- Every formula introduced should be followed by a sentence reading it back in words.
- When something is *not* derived (layer-norm gradient, autograd internals), say so explicitly and point to where it would be.
- Keep the inductive-bias thread visible: refer back to the table in section 9 whenever attention's "lack of built-in locality" comes up.
