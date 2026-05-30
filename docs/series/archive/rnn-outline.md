# RNN Post: Outline

## Title

**Recurrence: Time-Translation Equivariance and the Cost of a Bottleneck State**

## One-line thesis

A recurrent network's inductive bias is the assumption that *the same computation can be applied at every time step* and that *a fixed-size state can summarize the relevant past*. Time-translation equivariance is the symmetry; the bottleneck state is the price.

## Series position

Post 3 of 4. The thread:

- **Post 1 (scratchnn):** logits as the natural model output; loss as interpretation; backpropagation as composition of local steps.
- **Post 2 (CNN):** spatial translation equivariance. The same filter at every location. Locality plus weight sharing.
- **Post 3 (this post, RNN):** *temporal* translation equivariance. The same transition function at every step. Weight sharing along time.
- **Post 4 (Transformer):** abandon the recurrent bottleneck. Attention as content-addressable lookup across the whole past. Trained on multi-author text; this post's char-RNN is the obvious baseline.

Each architecture chooses a different group of symmetries to commute with. That is the unifying claim.

## Pedagogical arc

1. Start from the MLP. Note that for a sequence, a fixed-window MLP must pick a window length up front and re-learn the same pattern at every offset.
2. Introduce weight sharing along time. The MLP becomes an RNN: one cell, applied at every step, with a hidden state threading the steps together.
3. Derive the forward pass and BPTT carefully. Treat BPTT as backprop on the *unrolled* graph; no new chain rule, only composition through more layers (the same lesson as the MLP post).
4. Derive the vanishing-gradient problem from the recurrent Jacobian. The spectral radius of the recurrent Jacobian is what kills (or explodes) the gradient.
5. Show the inductive-bias picture: time-translation equivariance, summarized past, no parallelism across time.
6. Sketch LSTM and GRU as mitigations: gates carve a near-linear gradient highway through time.
7. Train on a single author's text. Show what a tiny RNN learns and where it fails.
8. Hand off to the Transformer post: attention drops the bottleneck and recovers parallelism.

## Section-by-section breakdown

### 1. From a window to a state

- A sequence model needs to map $x_1, x_2, \ldots, x_T$ to outputs $y_1, y_2, \ldots, y_T$ (or to a single $y$).
- An MLP applied to a fixed window $(x_{t-k}, \ldots, x_t)$ works but has two problems: the window is a hyperparameter, and a pattern learned at offset $j$ does not transfer to offset $j+1$.
- Weight sharing across time fixes both. Define a single transition function $f$ used at every step. The "memory" of past inputs lives in a hidden state $h_t$.

### 2. The vanilla RNN cell

- State and output equations:
  $$h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h),$$
  $$y_t = W_{hy}\, h_t + b_y.$$
- Parameters: $W_{hh} \in \mathbb{R}^{H \times H}$, $W_{xh} \in \mathbb{R}^{H \times D}$, $W_{hy} \in \mathbb{R}^{V \times H}$, biases $b_h \in \mathbb{R}^H$ and $b_y \in \mathbb{R}^V$. The *same* parameters appear at every $t$.
- Logits-vs-loss split carries over from post 1: $y_t$ is a logit vector; $\mathrm{softmax}$ and cross-entropy live in the loss, not the cell.
- Initial state $h_0$ is a hyperparameter or a learned parameter; we will set $h_0 = \mathbf{0}$.

### 3. Full unrolling

- Pick a sequence length $T$. The unrolled computation graph has $T$ copies of the cell with tied weights, plus $T$ output heads.
- Total loss for a sequence is $L = \sum_{t=1}^T L_t$, where $L_t$ is the per-step cross-entropy of $y_t$ against the target token at step $t$.
- Visually: the same column of operations stamped left-to-right, with one wire (the hidden state) running between columns. This is the moral picture for time-translation equivariance.

### 4. Backpropagation through time

- BPTT is backprop on the unrolled graph. No new chain rule, only more compositions.
- The seed at step $t$ is the familiar $p_t - y_t$ (post 1).
- Two upstream paths into $h_t$: from the output head $W_{hy}$ at step $t$, and from the next cell at step $t+1$ via $W_{hh}$:
  $$\frac{\partial L}{\partial h_t} = W_{hy}^\top (p_t - y_t) + W_{hh}^\top \mathrm{diag}(1 - h_{t+1}^2) \frac{\partial L}{\partial h_{t+1}}.$$
- Parameter gradients are sums over time of the local contributions. Weight sharing means the gradient of $W_{hh}$ is the sum of every step's contribution. (This is BPTT's only real twist over post 1's backprop, and it follows mechanically from the rule "the gradient of a shared parameter is the sum of its local gradients.")
- Truncated BPTT: in practice you backprop through a window of $\tau$ steps, then carry $h_t$ forward without gradient. Explain why (memory, gradient quality) and what the bias is (no credit to dependencies longer than $\tau$).

### 5. Why gradients vanish (or explode)

- The Jacobian of $h_{t+1}$ with respect to $h_t$ is
  $$J_t = \frac{\partial h_{t+1}}{\partial h_t} = \mathrm{diag}\bigl(1 - h_{t+1}^2\bigr)\, W_{hh}.$$
- Unrolling the chain over $k$ steps,
  $$\frac{\partial h_{t+k}}{\partial h_t} = \prod_{i=0}^{k-1} J_{t+i}.$$
- Bound the norm: $\bigl\| \partial h_{t+k} / \partial h_t \bigr\| \le \prod_i \|J_{t+i}\|$. If $\|J_{t+i}\| \le \gamma < 1$ for all $i$, the gradient decays at least as $\gamma^k$. If $\|J_{t+i}\| \ge \gamma > 1$, it grows at least as $\gamma^k$.
- Sharpen via spectral radius. Write $W_{hh} = U \Lambda U^{-1}$ (assume diagonalizable for the sketch). For small $h$, $\mathrm{diag}(1 - h^2) \approx I$, so $J_t \approx W_{hh}$ and $J_t^k \approx U \Lambda^k U^{-1}$. Components along eigenvectors with $|\lambda_i| < 1$ shrink as $|\lambda_i|^k$; those with $|\lambda_i| > 1$ blow up as $|\lambda_i|^k$. Even with $\tanh$ saturating the diagonal factor stays in $[0, 1]$, which only worsens the contraction. So $|\lambda_i| < 1$ forces vanishing, $|\lambda_i| > 1$ forces exploding.
- Consequence: long-range credit assignment is hopeless for vanilla RNNs at any depth where signal must propagate dozens of steps. Gradient clipping handles explosions cheaply; vanishing needs an architectural fix.

### 6. Inductive bias: time-translation equivariance and the bottleneck

- Equivariance statement: shift the input sequence by $k$ steps and the RNN's outputs shift by $k$ steps (modulo the initial state, which becomes a transient). The cell *commutes with time translation*.
- This is the same shape of claim as the CNN's spatial translation equivariance from post 2. CNNs share weights across space; RNNs share weights across time. Different group, same trick.
- The price: the entire past is squeezed through a fixed-dimensional vector $h_t$. This is the *Markov assumption with extra room*: $h_t$ must be a sufficient statistic of $x_{1:t}$ for predicting $x_{t+1:T}$. Any information the cell fails to write into $h_t$ is gone.
- Two consequences worth stating plainly:
  1. The model is forced to compress. That is good (it generalizes) and bad (rare long-range facts get dropped).
  2. Time-serial computation means no parallelism *within* a sequence at inference time. The Transformer post returns to this.

### 7. LSTM and GRU as gated variants

- The fix for vanishing gradients is structural: build a path along which the Jacobian is close to the identity by default.
- **LSTM** introduces a cell state $c_t$ updated as
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t,$$
  with forget gate $f_t = \sigma(W_f [h_{t-1}, x_t])$, input gate $i_t = \sigma(W_i [h_{t-1}, x_t])$, candidate $\tilde c_t = \tanh(W_c [h_{t-1}, x_t])$, output gate $o_t = \sigma(W_o [h_{t-1}, x_t])$, and hidden output $h_t = o_t \odot \tanh(c_t)$.
- The Jacobian $\partial c_t / \partial c_{t-1} = \mathrm{diag}(f_t)$ has eigenvalues in $[0, 1]$. When the network learns $f_t \approx 1$ for a coordinate, that coordinate's gradient flows back essentially unattenuated. The forget gate is the architecturally-provided gradient highway.
- **GRU** collapses the same idea into one fewer gate and no separate cell state. Reset gate $r_t$ and update gate $z_t$:
  $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t,$$
  $$\tilde h_t = \tanh(W_h [r_t \odot h_{t-1}, x_t]).$$
  The convex combination structure plays the same role as the LSTM's forget gate: gradients along the $(1 - z_t)$ branch get a near-identity Jacobian.
- The takeaway: gates are *differentiable selection*. They let the network choose, per coordinate and per step, whether to preserve, overwrite, or blend. The recurrent-equivariance bias is unchanged; only the gradient path is fixed.

### 8. Worked example

#### Corpus choice: Lewis Carroll, *Alice's Adventures in Wonderland*

Pick *Alice* (Project Gutenberg, public domain). Reasons:

- **Size.** About 144 KB / 27,000 words / 150,000 characters. Small enough to train in minutes on a CPU; big enough that a tiny model cannot memorize it verbatim.
- **Character set.** Plain ASCII with a small punctuation alphabet. Roughly 70 unique characters after lowercasing. No diacritics, no smart quotes after a light clean. Smaller alphabet means smaller softmax and a tractable demo.
- **Style distinctiveness.** Whimsical dialogue, frequent proper nouns ("Alice", "the Queen"), nonsense vocabulary ("curiouser", "Cheshire"), and short clause structure. The model has obvious tells to learn: capitalized names recur, dialogue follows quote marks and tags like *said the Cat*. The reader can *see* what the model has and has not picked up.
- **Why not the alternatives.** Shakespeare's sonnets are tiny (~95 KB) but verse structure rewards a model that captures meter, which a tiny vanilla RNN cannot. Austen is much longer (~700 KB) and prose-uniform, so progress is harder to read. Dickens (*A Christmas Carol*, ~170 KB) is a close runner-up; pick Carroll for the proper-noun signature and the slightly smaller alphabet.

The Transformer post (post 4) will scale to multi-author text and the per-author-style task becomes interesting there. Here we want a single voice the reader recognizes.

#### Target architecture

- **Vanilla 1-layer RNN.** Hidden dim $H = 128$. Sequence length $T = 64$ characters during training (truncated BPTT). Vocabulary $V \approx 70$.
- **Parameter count.** $W_{hh}: H \times H = 16{,}384$. $W_{xh}: H \times V \approx 9{,}000$. $W_{hy}: V \times H \approx 9{,}000$. Biases negligible. Total around 35,000 parameters. Small enough for pure Python on a 1 MB corpus to be at least tractable on CPU for a thousand or so updates.
- **Optimizer.** Plain SGD with gradient clipping (norm cap at 5). LR 0.1. Batch size 1 sequence; mini-batches of independent chunks add no math, only speed.
- **Comparison runs.** Same setup with an LSTM cell ($H = 128$) on the same data, same number of updates. Side-by-side samples make the gradient-flow argument concrete.

#### Expected generation quality

- **After ~100 updates (early):** uniform-ish noise, then character-frequency-correct gibberish (lots of "e", "t", spaces).
- **After ~1,000 updates:** bigram and trigram statistics; words like "the", "and", "she" appear; capitalization mostly wrong; quote marks unbalanced.
- **After ~10,000 updates:** plausible short words; "Alice" appears as a recurring token; sentence-like fragments with periods and capitals; dialogue tags begin to show up.
- **After ~50,000 updates:** locally coherent prose for two or three clauses; quote marks balanced over short spans; proper nouns mostly right; long-range coherence still absent (a paragraph wanders).
- **LSTM at the same step count:** noticeably better long-span balance (quotes, parentheticals), modestly better word boundaries. Not a different *kind* of model, just a model whose gradients survive long enough to learn dependencies that span dozens of characters.

These are *what the reader should expect*, not promises; the post should display actual samples at fixed checkpoints.

## Code approach recommendation

**Stay pure Python; do not introduce NumPy.** Justification:

- The scratchnn library's whole point is that backprop is small and visible. A `RNNCell` with hand-written forward and BPTT is a few dozen lines of the same flavor as `Linear`. Introducing NumPy here would erase the continuity with post 1.
- *Alice* at $T = 64$, $H = 128$, $V \approx 70$: each step is ~30K multiplies. 50,000 updates of length-$T$ sequences is ~$10^{11}$ floating-point operations. Pure Python at ~$10^7$ ops/sec finishes in a few hours. Workable for an overnight training run on a tiny corpus.
- The post's worked example can use a *much smaller* training slice (~10 KB, the first chapter only) to get the qualitative progression in minutes. Reserve the longer run for a generated artifact (a recording or a saved sample log) shipped with the post.
- The Transformer post is where NumPy pays for itself. Attention is $O(T^2)$ per layer, the matrices are bigger, and parallelizing across sequence positions is the whole point. Defer the NumPy switch to post 4, and make that switch part of *its* pedagogical argument: "here is the first architecture where you cannot get away with pure Python."

If the reader wants to actually train on the full *Alice* corpus, the post can include a sidebar that swaps the $W h$ products for `numpy.dot`, with everything else unchanged. That sidebar is also a clean ad for what an autograd library does for you.

## Figures to produce

1. **Loss curve.** Training cross-entropy per character vs update step, for the vanilla RNN and the LSTM on the same data. Log scale on the x-axis; the LSTM should pull ahead after a few thousand updates.
2. **Generation samples at fixed checkpoints.** A code block (or rendered grid) showing 200-character samples at update counts 100, 1k, 10k, 50k for both models. Same random seed for sampling; temperature 0.8.
3. **Gradient norm vs BPTT depth.** For a fixed batch, plot $\| \partial L / \partial h_{t-k} \|$ as a function of $k$ for $k = 0, 1, \ldots, 32$. Vanilla RNN: exponential decay. LSTM: a much flatter curve along the cell-state path. This is the visual proof of section 5 and the LSTM fix in section 7.
4. **Hidden-state PCA (optional, nice-to-have).** Feed the trained RNN a long sequence and collect $h_t$ at every step. Project to 2-D with PCA and color by something interpretable: inside-a-quoted-string vs not, vowel vs consonant context, recently-saw-Alice flag. Even a tiny RNN often shows a *quote indicator* dimension; if it does, that is the cleanest possible "the state really is summarizing the past" picture.
5. **Recurrent-Jacobian spectrum (optional).** Eigenvalue distribution of $W_{hh}$ at init vs after training. The trained vanilla RNN's spectral radius creeps up; the LSTM's $W_{hh}$ matrices look different by virtue of the gates around them.

## Handoff to the Transformer post

The RNN post should end by naming exactly what motivates attention:

1. **Long-range dependencies.** Even an LSTM struggles to credit information across hundreds of tokens at training scale. The cell-state highway helps; it does not eliminate the problem.
2. **The bottleneck state.** Everything the model knows about the past sits in one $H$-dimensional vector. Some questions ("what was the third word two paragraphs ago?") have no compressed answer in $h_t$; the information had to be discarded.
3. **No parallelism in time.** Inference cost is $O(T)$ serial steps. Training cost is the same. Modern hardware wants to do many things at once; a recurrent loop refuses.
4. **The setup for attention.** What if, instead of compressing the past into a single vector, every output step could *look up* whichever past steps are relevant? That is content-addressable memory, and it is what attention provides. The Transformer drops time-translation equivariance as a hard constraint (it becomes a soft positional encoding) in exchange for content-based routing and full sequence parallelism. Different symmetry budget, different trade.

The capstone post then trains an attention-based model on a multi-author corpus, where per-author style is the discriminative signal and the model's job is style-specific completion. The RNN baseline from this post becomes the "what we used to do" reference.

## Open questions for the author

- Should the LSTM derivation be a sidebar or a section? Current outline treats it as a real section because the gates' role in gradient flow is the central payoff. If we cut it to a sidebar, section 5's vanishing-gradient argument loses its resolution.
- Worth shipping a tiny `rnn.py` module alongside the post, or keep the post pure prose-and-math? Argument for a module: it would be ~150 lines and would *literally* extend the scratchnn idiom (a new `RNNCell` layer with a `forward(x_t, h_{t-1})` and `backward(g_h, g_y)`). Argument against: scratchnn's `Layer` protocol assumes a single-input-vector forward; recurrence breaks that and might require a small protocol extension. Decide before drafting.
- Carroll vs Dickens? Both work. Carroll preferred for proper-noun signature; Dickens preferred if we want the slightly more grown-up prose register for the Transformer comparison in post 4.
