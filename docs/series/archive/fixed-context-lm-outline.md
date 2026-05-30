# Fixed-Context Language Model Outline (Post 4 of 7)

## Title and thesis

**Title:** A Neural Probabilistic Language Model in 200 Lines: Bengio 2003 from Scratch

**One-line thesis:** A language model is a next-token classifier, and the simplest neural one is `scratchnn`'s softmax regression with one twist: the input is the concatenation of the last $n$ token embeddings. Bengio et al. (2003) is the historical reference; everything since (RNNs, Transformers) is a story about lifting the fixed-window assumption.

## Series position

Post 4 of 7. The thread:

- **Post 1 (scratchnn walkthrough):** logits as model output, loss as interpretation, backprop as composition.
- **Post 2 (output heads):** the loss is a link function; choosing it is choosing what we believe about the response.
- **Post 3 (CNN):** spatial inductive bias via weight sharing.
- **Post 4 (this post, fixed-context LM):** sequence prediction as the new task; embeddings, the softmax head over the vocabulary, and the cost of the fixed window.
- **Post 5 (RNN):** lift the fixed window with a recurrent state.
- **Post 6 (Transformer):** lift the bottleneck state with attention.
- **Post 7 (RL):** lift the supervised assumption itself; policy gradients and a forward-pointer to AIXI.

This post is the hinge between the regression/classification thread (posts 1 to 3) and the sequence-modeling thread (posts 5 to 7). It pays off the framing that *every* loss in the series is interpreting logits, and it introduces the only architectural prior the rest of the series will keep dismantling: the bounded context.

## Pedagogical arc

The reader should leave with these, in order:

1. **Sequence prediction is the universal learning problem.** Solomonoff's incomputable optimum tells us what the "right" answer would be: weight every program that could have produced the past by $2^{-\ell}$. Every concrete model in this series is a *computable* slice of that.
2. **Cross-entropy is bits per token.** MDL is not an analogy. Training the LM minimizes the same quantity a compressor would.
3. **The corpus labels itself.** No annotation budget, no labeled data. Self-supervised learning is just "input and target are both extracted from the same sequence."
4. **Embeddings are a `Linear` layer with a one-hot input.** Same math, more efficient bookkeeping.
5. **The Bengio architecture is one new layer (`Embedding`) and a familiar `Linear -> Tanh -> Linear -> SoftmaxCrossEntropy` head.** The chain rule does not change. Backprop into the embedding table is the `Linear`-with-one-hot gradient, written without instantiating the one-hot.
6. **The fixed window is a ceiling.** As $n$ stays fixed, perplexity plateaus. The model literally cannot use information older than $n$ tokens. That ceiling is what motivates the rest of the series.

## Section-by-section breakdown

### 1. The task: predict the next token

- A corpus is a sequence $x_1, x_2, \ldots, x_N$ over a vocabulary $\mathcal{V}$ of size $V$.
- The model is a conditional distribution $p_\theta(x_{t+1} \mid x_{1:t})$.
- Training loss is the negative log-likelihood of the corpus under $p_\theta$:
  $$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{t=1}^{N-1} \log p_\theta(x_{t+1} \mid x_{1:t}).$$
- Per-token cross-entropy in nats. Divide by $\log 2$ for bits. Exponentiate for **perplexity** $\mathrm{PPL} = \exp(\mathcal{L})$, the geometric-mean inverse probability the model assigns to the truth.
- This is not a new loss. It is `SoftmaxCrossEntropy` from the walkthrough, evaluated at every position.

### 2. Solomonoff induction: the incomputable target

- Imagine the ideal next-token predictor. It would consider every program $q$ that, run on a universal Turing machine, outputs a prefix matching $x_{1:t}$, weight it by its prior $2^{-\ell(q)}$ (shorter program, larger prior, an Occam choice), and use the induced predictive distribution:
  $$\xi(x_{t+1} \mid x_{1:t}) \propto \sum_{q\,:\,U(q) \text{ starts with } x_{1:t} x_{t+1}} 2^{-\ell(q)}.$$
- This is the Solomonoff prior. It is the dominant universal semi-measure: for any computable distribution $\mu$, $\xi(x_{1:n}) \ge 2^{-K(\mu)} \mu(x_{1:n})$, where $K(\mu)$ is the (constant) Kolmogorov complexity of the program for $\mu$. Solomonoff's theorem says posterior predictions under $\xi$ converge to those of the true $\mu$ with finite total expected squared error.
- It is incomputable. We cannot enumerate Turing machines and check halting. But it sets the meta-question every sequence model is approximately answering: *what is a good prior over programs that produce sequences?*
- Every neural architecture commits to a different computable family of programs and a different inductive bias over that family. An $n$-gram counts subsequences. A fixed-context MLP picks one mapping per window. An RNN constrains its program to "Markov state plus transition." A Transformer's program is "attention over the past." None is universal; each is a tractable slice.
- This frames the rest of the post: we are choosing a function class, and we should be honest about which class.

### 3. MDL: cross-entropy *is* bits

- Minimum description length: pick the model minimizing
  $$\mathrm{bits}(\text{model}) + \mathrm{bits}(\text{data} \mid \text{model}).$$
- The second term, with a probability model $p_\theta$ and an optimal code (Shannon), is exactly
  $$-\sum_t \log_2 p_\theta(x_{t+1} \mid x_{1:t}).$$
- That is the training loss in bits. Not an analogy. The same number.
- Consequence: if model A achieves lower cross-entropy on held-out text than model B, A is a strictly better compressor of that text. Generalization and compression are two scorings of one quantity.
- The MDL view also rationalizes regularization (the $\mathrm{bits}(\text{model})$ term). We are *not* adding regularization here, but the framing pre-explains why it would appear: tradeoff between description length of weights and code length of data.
- Tie to Solomonoff: the universal prior is the unreachable limit where $\mathrm{bits}(\text{model})$ is the literal Kolmogorov complexity. A neural network parameterizes a tractable family and pays $\mathrm{bits}(\text{model})$ in floating-point storage.

### 4. Self-supervised learning, named

- The corpus is the data and the labels. For every position $t$, the past $x_{1:t}$ is the input, the next token $x_{t+1}$ is the target. No annotation; the structure of the sequence supplies the supervision.
- Mathematically identical to supervised learning. The same `SoftmaxCrossEntropy` loss, the same gradient, the same `fit` loop.
- Practically transformative: the data supply is effectively unbounded (any text), and every position contributes a gradient signal. A 1 MB book yields ~$10^6$ training examples for free.
- Modern usage of "self-supervised" covers contrastive learning (SimCLR, CLIP), masked autoencoding (BERT, MAE), and span-prediction variants. Mention them as the same idea generalized: the task is constructed from the data itself.
- For this post the framing is just: next-token prediction is the canonical SSL task, and language modeling is its dominant instantiation.

### 5. Tokenization

- The model operates on tokens, not raw text. Choosing the vocabulary is choosing the model's atomic units.
- **Character-level** (what this post uses). $V \approx 70$ for cleaned ASCII English. Pros: small vocabulary, trivial to implement, transparent to debug, no out-of-vocabulary problem. Cons: long sequences (a single word is many tokens), the model must learn that letters form words.
- **Word-level.** Large $V$ (tens of thousands), OOV problem, but short sequences. Standard before BPE.
- **Subword (BPE, WordPiece, Unigram).** Forward-pointer for the Transformer post. Compromise: learned vocabulary of common substrings; rare words decompose into pieces. Modern default.
- For this post: character-level on *Alice* (same corpus the RNN post uses). The continuity matters; the reader will be comparing perplexity numbers in post 5.

### 6. Embeddings: a `Linear` with a one-hot input

- A token id $v \in \{0, \ldots, V-1\}$ is not a useful input. It is a category label. The model needs a vector representation.
- One choice: one-hot encode $v$ as $\mathbf{e}_v \in \mathbb{R}^V$ (the $v$-th standard basis vector). Then pass $\mathbf{e}_v$ through a `Linear(V, d)`. The output is exactly column $v$ of the weight matrix.
- That is an embedding. Define an embedding table $E \in \mathbb{R}^{V \times d}$ and let $\mathrm{embed}(v) = E[v]$, the $v$-th row.
- Embedding is `Linear` with a one-hot input. Same math, two efficiency wins:
  1. We never materialize the $V$-dimensional one-hot.
  2. The forward pass is a single array lookup, $O(d)$ instead of $O(V d)$.
- **Backward.** With $g \in \mathbb{R}^d$ the gradient at the embedding output,
  $$\frac{\partial L}{\partial E[v]} \mathrel{+}= g, \qquad \frac{\partial L}{\partial E[v']} = 0 \text{ for } v' \ne v.$$
  Only the row used in the forward pass gets a gradient. Over a mini-batch the gradient at row $v$ accumulates the per-example gradients from every position that used token $v$. The `+=` pattern is the same one `Linear` and `Conv2D` already use, restricted to one row.
- This is one of the post's load-bearing moments. The embedding table is a parameter, gradients flow through it, and the only new bookkeeping is "which row was selected." Everything else is `Linear`.

### 7. The Bengio 2003 architecture

- Inputs: the last $n$ token ids $(v_{t-n+1}, \ldots, v_t)$. Call this the context window.
- Forward pass:
  1. Embed each context token: $\mathbf{e}_i = E[v_i]$ for $i = t-n+1, \ldots, t$. Each $\mathbf{e}_i \in \mathbb{R}^d$.
  2. Concatenate into a single window vector $\mathbf{u} = [\mathbf{e}_{t-n+1}; \ldots; \mathbf{e}_t] \in \mathbb{R}^{n d}$.
  3. MLP: $\mathbf{h} = \tanh(W_1 \mathbf{u} + \mathbf{b}_1)$ with $W_1 \in \mathbb{R}^{H \times n d}$.
  4. Output: $\mathbf{z} = W_2 \mathbf{h} + \mathbf{b}_2$ with $W_2 \in \mathbb{R}^{V \times H}$. The logits.
  5. Loss: `SoftmaxCrossEntropy(z, v_{t+1})`.
- Bengio's original paper adds a direct skip connection from $\mathbf{u}$ to $\mathbf{z}$ (a separate $W_3 \mathbf{u}$ added to $\mathbf{z}$). Mention as a footnote; omit from the worked example to keep the architecture clean.
- This is `scratchnn` plus one new layer:
  ```python
  net = Network([
      Embedding(V, d),     # the only new piece
      Concat(n, d),        # trivial: stack n rows into one vector
      Linear(n*d, H), Tanh(),
      Linear(H, V),
  ], loss=SoftmaxCrossEntropy())
  ```
  `Concat` is a one-line layer: forward concatenates, backward splits.
- Parameter count: $\underbrace{V d}_{E} + \underbrace{n d \cdot H + H}_{W_1, \mathbf{b}_1} + \underbrace{H V + V}_{W_2, \mathbf{b}_2}$. For $V = 70$, $d = 16$, $n = 8$, $H = 64$: $1120 + 8256 + 4550 = 13{,}926$. Small enough to train in pure Python.

### 8. Gradient flow and the familiar $\mathbf{p} - \mathbf{y}$

- The loss gradient at the logits is the walkthrough's twin:
  $$\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}, \qquad \mathbf{p} = \mathrm{softmax}(\mathbf{z}), \qquad \mathbf{y} = \mathrm{onehot}(v_{t+1}).$$
- That gradient flows back through `Linear`, `Tanh`, `Linear`, `Concat` (splits the vector into $n$ pieces of length $d$), and finally into `Embedding`, where each piece accumulates into the corresponding row of $E$.
- Nothing in this chain is new. The new piece, embedding, is a routed gradient: row $v$ of $E$ receives the gradient destined for the embedding output, and other rows receive nothing.
- The training loop is `net.fit(X, Y, ...)` where $X[t] = (v_{t-n+1}, \ldots, v_t)$ and $Y[t] = v_{t+1}$.

## Worked example: Alice on 8-char windows

### Corpus

- *Alice's Adventures in Wonderland* (Project Gutenberg), the same corpus the RNN post uses. About 150 KB after stripping the Gutenberg header/footer.
- Light cleaning: lowercase, collapse whitespace, drop characters not in the ASCII letter/punctuation set. Vocabulary $V \approx 65$ to $70$.
- Train/test split: first 90% for training, last 10% for held-out perplexity. Mention but do not dwell on the temporal split assumption.

### Architecture

- $n = 8$ characters of context, $d = 16$ embedding dimension, $H = 64$ hidden units. One `Tanh` hidden layer.
- Parameters: ~14k (see section 7).
- Training: SGD, batch size 32, learning rate 0.05, 20 epochs over the corpus. A few hours on a CPU in pure Python; acceptable for an overnight run.

### Expected results

- **Per-character cross-entropy on held-out text:** target around 2.0 to 2.4 nats, perplexity 7 to 11. Unigram baseline (just character frequencies) is around 2.8 nats. A 4-gram count model with backoff would be in the same ballpark as the neural model at $n = 8$; the neural model wins as $n$ grows because count tables blow up.
- **Generated samples** (greedy or temperature-0.8 sampling, conditioned on a seed):
  - **Epoch 1:** "the the the the and the and" or character-frequency-correct noise.
  - **Epoch 5:** real short words appear ("the", "and", "she"), spacing improves, longer "words" are still gibberish.
  - **Epoch 20:** plausible English-shaped strings, proper nouns sometimes recognizable ("alice"), short phrases coherent. Long-range coherence absent. Quote marks unbalanced. A sentence trails off into syllabic mush.
- The "gibberish to local coherence" arc is the dramatic figure. Same shape as the RNN post will show. The difference is the *ceiling*: this model cannot improve past what 8 characters of context allows.

### The ceiling experiment

- Train the same architecture at $n \in \{2, 4, 8, 16, 32\}$. Same $d$, same $H$, same training budget.
- Plot held-out perplexity vs $n$. Expected: monotone decrease, but with diminishing returns. By $n = 32$ the parameter count is dominated by $n d \cdot H = 32 \cdot 16 \cdot 64 = 32{,}768$, the model is bigger, and the gains are small. The curve flattens.
- The information-theoretic interpretation: held-out perplexity is bounded below by the entropy of the true conditional distribution given $n$ tokens of context, $H(p^*_n)$. As $n \to \infty$, $H(p^*_n) \to H(p^*)$, the per-character entropy of English. Empirically that limit is around 1.0 to 1.5 nats per character (Shannon's "Prediction and Entropy of Printed English"). Our model sits well above this floor because (a) it has a fixed window and (b) it has finite capacity. The fixed window contribution is what we are isolating.
- This is the experiment that *motivates* the RNN post. The reader sees a wall and is told the next post climbs it.

## Math content checklist

- Cross-entropy in nats and bits; perplexity. The connection $\mathrm{PPL} = \exp(\mathcal{L})$.
- Embedding gradient as `Linear`-with-one-hot. The row-selection form $\partial L / \partial E[v] = g$ for the selected row, zero elsewhere; accumulation over positions and batch.
- Softmax cross-entropy gradient $\mathbf{p} - \mathbf{y}$ (recall from walkthrough; no re-derivation needed).
- Parameter count $V d + n d H + H V$. Three terms: vocabulary $\times$ embedding, window $\times$ hidden, hidden $\times$ vocabulary. The window term grows linearly in $n$; the others are fixed.
- Solomonoff dominance: $\xi(x_{1:n}) \ge 2^{-K(\mu)} \mu(x_{1:n})$. Stated, not derived.
- MDL identity: cross-entropy in bits $=$ length of the Shannon code for the data under $p_\theta$. One sentence each.
- Information-theoretic ceiling: $\mathcal{L}_n \ge H(p^*_n) \ge H(p^*)$, where $\mathcal{L}_n$ is the best achievable cross-entropy at context length $n$.

## Library additions

The post requires two new pieces in `scratchnn`. Both fit the `Layer` protocol; neither requires changes to `Network`, `zero_grad`, or `step`.

### `EmbedConcat(V, d, n)`

The input layer for the worked example. Takes a list of $n$ token ids, looks up each row of $E$, and concatenates into a single vector of length $n d$.

```python
class EmbedConcat(Layer):
    """Embed n token ids and concatenate into one vector of length n*d."""
    def __init__(self, V, d, n):
        r = 1.0 / math.sqrt(d)
        self.V, self.d, self.n = V, d, n
        self.E = [[random.uniform(-r, r) for _ in range(d)] for _ in range(V)]
        self.dE = [[0.0 for _ in range(d)] for _ in range(V)]
        self.ids = None

    def forward(self, ids):
        self.ids = ids
        out = []
        for v in ids:
            out.extend(self.E[v])
        return out

    def backward(self, g):
        for k, v in enumerate(self.ids):
            for j in range(self.d):
                self.dE[v][j] += g[k * self.d + j]
        return None  # no gradient back into categorical inputs

    def parameters(self):
        return list(zip(self.E, self.dE))
```

About 20 lines. `parameters()` yields one `(row, dRow)` pair per vocabulary entry; `zero_grad` and `step` work generically on those flat-list pairs, exactly as for `Linear`.

A standalone `Embedding(V, d)` whose forward takes a single `int` is the cleaner pedagogical object (the literal "`Linear` with a one-hot input"), but it departs from the protocol's "forward takes a vector" assumption. The post should derive `Embedding` mathematically and then introduce `EmbedConcat` as the practical fusion the worked example uses.

### Tokenizer helper

A `tokenize.py` (or a section of `examples/`) with three functions:

```python
def build_vocab(text):        # -> (char -> id, id -> char)
def encode(text, char_to_id): # -> list[int]
def decode(ids, id_to_char):  # -> str
```

Plain dicts and list comprehensions. No new abstractions.

### Gradient checks

Add `gradient_check` cases for `Embedding` and `EmbedConcat`. Pick a tiny $V$ and $d$, fixed random ids, finite-difference the embedding rows. The kink caveat does not apply (lookup is everywhere differentiable in the parameters).

## Code approach: stay in pure Python

Same call as the CNN post. Justifications:

- The whole pedagogy depends on the reader being able to read the library. Adding NumPy here breaks the chain.
- 14k parameters and ~150k training examples (one per position) on a 70-character vocabulary is feasible in pure Python. Forward pass per example: $n d + n d H + H V \approx 128 + 8192 + 4480 \approx 13$k multiplies. Backward roughly the same. 20 epochs $\times$ 135k examples $\times$ ~30k multiplies is around $10^{11}$ operations, hours not days on a CPU.
- Ship the trained model's outputs (loss curves, sample text at checkpoints) as artifacts so the reader does not need to reproduce the run. The pure-Python code is for *reading*, not for the casual reader to retrain.
- The Transformer post is where the dam breaks. Attention is $O(T^2 d)$ per layer, vocabulary into the tens of thousands, and the parallelism story is the whole point. Defer the NumPy switch to post 6 (or to the RNN post if its run times push us there first).

## Inductive bias framing

The fixed-context LM commits to one strong prior:

- **Only the last $n$ tokens matter.** Any structure across a longer span is invisible to the model. Subject-verb agreement across a clause, pronoun resolution across a sentence, a callback to "Alice" two paragraphs ago: all impossible by construction at $n = 8$.

What is given up:

- Arbitrary-history dependence. To get more context, pay more parameters: $W_1$ grows linearly in $n$. Doubling the window doubles the input layer's weight count. The architecture trades parameter count for context length, *one for one*.
- Time-translation equivariance. Each position in the window has its own slice of $W_1$. A pattern learned at offset 3 of the window must be re-learned at offset 4. The CNN's spatial-equivariance prior is exactly what we are *not* using here, and the RNN's temporal-equivariance prior is what we will introduce next.

What is gained:

- Parallelism. Every position's prediction is independent of every other given the corpus; we can shuffle training examples freely.
- A genuinely simple architecture. The only new mathematical object is the embedding table. Everything else is the `scratchnn` from post 1.
- An honest baseline. The Bengio 2003 architecture is the simplest neural language model that works at all. Improvements on top of it (RNN, Transformer) are easier to evaluate when the baseline is this clear.

The inductive-bias contract is the headline: a fixed window is an *axiom* the model cannot question. The next two posts each replace that axiom with a different one. This post earns the reader the right to compare.

## Figures to produce

Five figures, ranked by load-bearing:

1. **Sample text at epochs 1, 5, 10, 20.** The gibberish-to-local-coherence arc, four blocks of generated text from the same seed prompt at fixed checkpoints. This is the headline figure: the model visibly learns.
2. **Perplexity vs window size $n$.** Five points: $n \in \{2, 4, 8, 16, 32\}$. Held-out per-character cross-entropy on the y-axis. The curve flattens; the visual proof of the ceiling. This is the figure that *motivates* the RNN post.
3. **Loss curve at the headline configuration.** Training and held-out cross-entropy vs epoch for $n = 8$. Standard training-diagnostic figure. The held-out curve plateaus first; demonstrate generalization.
4. **Embedding PCA (optional, nice-to-have).** Project the trained $E$ to 2-D with PCA, label points with their characters. Often vowels cluster, punctuation clusters, capital and lowercase pair up. The cleanest possible "the network learned representations" picture.
5. **Embedding gradient sparsity (optional).** Heatmap of $\|\partial L / \partial E[v]\|$ for one mini-batch. Most rows are zero (their tokens did not appear). The visual is striking and reinforces the "one-hot input means one-row gradient" point from section 6.

If only one figure makes the cut: figure 1. If two: 1 and 2.

## Handoff to the RNN post

The fixed-context LM has hit a ceiling that has nothing to do with optimization, dataset size, or model width. It is a ceiling imposed by the architecture: no information older than $n$ tokens can enter the prediction. Doubling $n$ doubles $W_1$. Going to $n = 1000$ doubles $W_1$ ten more times.

The RNN drops the window. It introduces a hidden state $h_t$ that summarizes $x_{1:t}$ for prediction, with a single transition function applied at every step. The same parameters every position, the way the CNN's kernel is the same parameters at every spatial location. We trade the fixed-window prior for two new commitments:

1. A fixed-size state can summarize the relevant past. (Bottleneck.)
2. The same computation works at every time step. (Time-translation equivariance.)

Both are wrong in detail, both are useful in practice, and post 5 shows what each costs.

The corpus is the same: *Alice* at character level. The reader can read post 4's samples and post 5's samples side by side. The comparison is the point.

## Open questions for the author

- **Embedding as a `Layer` whose forward takes an `int`.** The current `Layer` protocol assumes `forward(x)` where `x` is a vector. `Embedding`'s forward takes an `int`. Two paths: (a) document the type departure and accept it; (b) introduce `EmbedConcat` as the only embedding-flavored layer and never expose the int-input form. The outline recommends (b) for the worked example and (a) for the pedagogical mention. Decide before drafting.
- **Skip connection from $\mathbf{u}$ to $\mathbf{z}$.** Bengio's original architecture has one. It improves perplexity modestly and adds a third weight matrix. Recommend: footnote only, omit from the worked example. Worth a 30-second mention.
- **Where to put the Solomonoff and MDL sections.** Currently sections 2 and 3, before the architecture. Risk: reader bounces if the theory is too far in front of the code. Alternative: open with the task (section 1), introduce the architecture (sections 5 to 7), then drop the Solomonoff/MDL framing as a "what were we actually doing?" reflection after the worked example. The current order is the more honest one but the harder one to keep moving. Author's call.
- **Compare to an $n$-gram baseline?** A Kneser-Ney smoothed 4-gram is roughly competitive with this model on a small corpus. Including it would sharpen the "what does the neural part buy us?" question, but it adds a section. Recommend: one paragraph, one number, no code.
- **Where the post lives in the repo.** The library additions (`Embedding`, `EmbedConcat`, tokenizer helpers) belong in `scratchnn` if and only if subsequent posts (RNN, Transformer) also extend the library. If those posts stay prose-and-math without code, the additions could live under `examples/lm/` instead. Recommend extending the library: the RNN post is already considering a `RNNCell`, and the Transformer post will want a vocabulary-aware head. Decide in concert with the RNN-post author.
- **Length budget.** This is the first post in the series with three theoretical openers (Solomonoff, MDL, SSL), an architecture section, a worked example, and a ceiling experiment. The cnn-outline and rnn-outline siblings ran ~150 to 180 lines of outlined post-content; this post may be 30% longer when written. Consider splitting the ceiling experiment into a sidebar to keep the main thread tight.
