# Design: Part I (Foundations) of Inductive Biases in Neural Networks

**Author:** Alexander Towell
**Date:** 2026-06-02
**Status:** Design (pre-implementation)
**Parent spec:** `docs/superpowers/specs/2026-06-02-master-design.md`
**Scope:** Section-level outline for Chapters 1 and 2 (Part I, ~22 pp). Authoritative for the Part I implementation plans.

## 1. Purpose

Part I builds the floor that the rest of the book stands on, and it introduces the book's thesis. Chapter 1 develops the network as a parameterized function approximator, shows why a single linear unit cannot separate XOR, recovers the multilayer perceptron as the smallest fix, and derives backpropagation as the local chain rule. Chapter 2 takes the first inductive bias proper, the output head, and proves the canonical-link theorem that explains why the `p - y` gradient recurs.

By the end of Part I the reader can build an MLP from a three-method `Layer` protocol, can explain the two independent axes at which a prior enters a supervised model (output head and architecture), and can choose an output head for real-valued, binary, categorical, count, or heteroscedastic targets and justify the choice as a prior. The two axes that organize the rest of the book both make their first appearance here.

This Part is a re-homing of the existing booklet chapters (`../../docs/booklet/chapters/01-foundations.tex`, `02-output-heads.tex`) and their series originals. The prose largely exists. The work is porting it to hand-authored LaTeX, verifying every listing against the live library, regenerating every number from a notebook, and removing the blog-serialization artifacts.

## 2. Inherited Commitments (from the master spec)

- Audience: the self-study practitioner. MacKay-tradition monograph, intuition first, hand-derived math, real numbers. No exercises, no learning-outcome boxes.
- Voice: single voice, soul rules on `.tex` (no em-dashes; commas, colons, periods, parentheses).
- Real-code policy: every listing is taken from `../../src/scratchnn`, lightly trimmed (init boilerplate elided, hand-derived `backward` kept in full). Never pseudocode.
- Citation: per-chapter Bibliographic Notes resolving to `book/references.bib` (biblatex/biber).
- Notebooks: one executed notebook per chapter that makes live numerical claims. Both Part I chapters qualify.

## 3. Decisions Settled for Part I

Three choices, settled in the Part I design dialogue, apply across both chapters and (for the first) the whole book.

**(A) LaTeX auto-numbering with `\label` and `\Cref`.** Sections are authored as `\section{Title}` with no manual number in the title. Each section carries a `\label` (scheme in Section 4). All cross-references use `\Cref{}`. This replaces the booklet's pandoc-era authored numbers ("1. What is a neural network?") and prose references ("section 7"). It is the bookwright cross-reference discipline and it is the natural form for hand-written `.tex`. It is a book-wide convention; Part I is where it starts.

**(B) Full de-serialization.** Beyond fixing outright ordering errors, Part I converts all blog idiom ("this post", "follow-up post") to chapter language, reconciles the two-axes overlap between `\Cref{sec:inductive-biases}` (Ch 1) and `\Cref{sec:parallel-axes}` (Ch 2) so Chapter 1 previews and names the axes while Chapter 2 owns the full synthesis, and reframes the "reader exercise" pointers (Beta NLL, MDN) as "natural extensions" in the manner of Chapter 1's autograd closing note (consistent with the no-exercises decision).

**(C) Notebook regenerates the numbers.** The paired notebook is the source of truth for every numerical claim. Seeds are fixed and recorded; prose numbers are written to match the notebook's actual output. This applies most sharply to Chapter 2's specific values (Poisson minimum-rate 0.068 vs 0.310; the heteroscedastic prediction table; test Gaussian NLL 0.745 vs 0.509; gradient-check residuals near 1e-11).

## 4. Library State (verified, supersedes the master spec's stale note)

The master spec said the library implemented only two output heads. That cited the older `docs/design.md`. The live `src/scratchnn/__init__.py` `__all__` exports, verified for this spec:

```
dot, sigmoid, logsumexp, softmax, softplus,
Layer, Linear, Tanh, ReLU,
Loss, SigmoidBCE, SoftmaxCrossEntropy, MSELoss, PoissonNLLLoss, GaussianNLLLoss,
Conv2D, GlobalAvgPool, RNNCell, Embedding, EmbedConcat,
Network, gradient_check
```

So Chapter 2 already has real, hand-derived code for all five heads it discusses: `MSELoss` (Gaussian fixed variance), `SigmoidBCE` (Bernoulli), `SoftmaxCrossEntropy` (Categorical), `PoissonNLLLoss` (Poisson), `GaussianNLLLoss` (heteroscedastic Gaussian), plus the `softplus` helper. Beta NLL and the MDN are intentionally absent and become natural-extension pointers. The master spec is patched accordingly; there is no output-heads gap to close.

## 5. Chapter 1: Foundations (~11 pp, has notebook)

Title (canonical, from the booklet; drops the series' "Classification" and "Multi-Layer"): **Foundations: Function Approximation and the Multilayer Perceptron**. Chapter label `\label{ch:foundations}`.

All sections port from booklet ch1 plus series 01. Code listings come from `src/scratchnn/neural_net.py`. "Source" below names the specific symbol or experiment, or marks fresh prose.

| § | Title (`\label`) | pp | Source | Purpose |
|---|---|---|---|---|
| 1 | What is a neural network? `sec:nn-as-function` | 1.0 | `MSELoss` | Supervised setup; f_theta : R^n to R^m; the output's meaning is the loss's job. Identity + MSE as the unspecialized base case. First sighting of the residual gradient (y-hat minus y). |
| 2 | The single linear unit and its limits `sec:linear-limits` | 1.5 | `SigmoidBCE` on XOR (fails) | SLP is an affine map. XOR is not linearly separable: the algebraic contradiction and the diagonal-corners geometry. Minsky and Papert 1969. Motivates the non-linearity. |
| 3 | The multilayer perceptron, and what depth and non-linearity buy `sec:mlp-backprop` | 2.0 | `Layer`, `Linear`, `Tanh`, `ReLU`, `Network` | Stacking with Tanh/ReLU; collapse without non-linearity; universal approximation (Cybenko, Hornik); backprop as the local chain rule. The three-method `Layer` protocol and the once-written generic `step`/`zero_grad`. "No new math, only composition." |
| 4 | Logistic regression `sec:logistic` | 1.0 | `SigmoidBCE` | Logit, sigmoid, BCE; the clean `p - y` gradient. One `Linear` + `SigmoidBCE`. The head owns the activation; the network emits raw logits. |
| 5 | Multi-class regression via softmax `sec:softmax` | 1.0 | `SoftmaxCrossEntropy`; UCI digits | K logits, softmax, cross-entropy; `p - y` with one-hot y. Binary logistic is two-class softmax with one logit pinned. UCI 8x8 digits: ~95% (softmax) and ~96% (MLP). The "why is the gap so small?" hook forward to `\Cref{sec:inductive-biases}`. |
| 6 | Logits are log-probabilities, up to a constant `sec:logits-logprobs` | 1.0 | fresh | log softmax = z minus logsumexp; logits are unnormalized log-probabilities; the additive constant is gauge (unobservable); training pins only log-odds differences. The ReLU-on-logits argument. |
| 7 | Inductive biases `sec:inductive-biases` | 1.5 | booklet ch1 §7 | THE THESIS. Names the two axes (output head, architecture) and the MLP's independence prior. De-serialize: previews the axes and keeps the one-number CNN-vs-MLP parameter-count teaser (~160 vs ~2500); defers the full composability synthesis to `\Cref{sec:parallel-axes}`. "Follow-up post" becomes `\Cref{ch:output-heads}`. |
| 8 | Numerical stability is the gauge freedom, reused `sec:stability` | 1.0 | `sigmoid`, `logsumexp`, `softmax` helpers | Overflow; max-subtraction is exactly the `\Cref{sec:logits-logprobs}` additive-constant freedom spent for safety. logsumexp; sign-branched sigmoid. Stability is the gauge symmetry put to work, not a bolt-on. |
| 9 | Trust, but verify: numerical gradients `sec:gradcheck` | 1.0 | `gradient_check` | Central difference; the checker that touches only `parameters()`/`forward`/`backward`; the ReLU-kink caveat (tolerance anchored on Tanh). |
| 10 | Closing note: from per-layer to per-operation `sec:autograd-note` | 0.5 | fresh | Per-operation local backward is automatic differentiation; the natural generalization, pointed at, not built. The template for Chapter 2's MDN and Beta pointers. |

### Chapter 1 Notebook (`notebooks/ch01-foundations.ipynb`)

- XOR: the single `Linear` + `SigmoidBCE` model converges to p ~ 0.5 (cannot separate); the 2-8-1 Tanh MLP fits it. Regenerate the decision-boundary figure.
- The sin-regression demo (identity + MSE) used in `\Cref{sec:nn-as-function}`.
- UCI digits: softmax ~95%, 64-32-10 MLP ~96%, train fit ~99.95%. These numbers feed `\Cref{sec:softmax}` and the `\Cref{sec:inductive-biases}` hook.
- `gradient_check` residuals across logistic, softmax, Tanh-MLP, ReLU-MLP configurations (near 1e-10).
- The CNN ~160-weight vs MLP ~2500-weight parameter-count claim (computed, not asserted) for the teaser in `\Cref{sec:inductive-biases}`.
- Pure Python plus the quarantined `visualize.py` for the XOR boundary; no NumPy.

## 6. Chapter 2: Output Heads as Inductive Bias (~10 pp, has notebook)

Title: **Output Heads as Inductive Bias**. Chapter label `\label{ch:output-heads}`. Sections port from booklet ch2 plus series 02; loss listings from `src/scratchnn/neural_net.py`.

| § | Title (`\label`) | pp | Source | Purpose |
|---|---|---|---|---|
| 1 | The unifying frame `sec:mle-frame` | 0.75 | booklet ch2 §1 | Every supervised loss is an NLL; the network parameterizes p(y given x); the head is link plus assumed distribution. Ties to Chapter 1's "network emits logits, loss interprets them" via `\Cref{ch:foundations}`. |
| 2 | The catalogue `sec:catalogue` | 1.0 | booklet ch2 §2 | Five canonical pairings (Gaussian, Bernoulli, Categorical, Poisson, Gamma) with canonical links. The link maps the unconstrained output to a valid distribution parameter. |
| 3 | The canonical link theorem `sec:canonical-link` | 1.5 | fresh derivation | Exponential family natural form; A'(eta) = E[T(y)]; with the canonical link the NLL gradient is (p-hat minus y). Proves Chapter 1's "no coincidence" `p - y` claim (`\Cref{sec:logistic}`, `\Cref{sec:softmax}`). |
| 4 | Chapter 1's three pairings, named `sec:three-pairings` | 1.0 | booklet ch2 §4 | Recap identity/Gaussian/MSE, logit/Bernoulli/BCE, softmax/Categorical/CE, each cross-referenced back into Chapter 1. One pattern, three objects. |
| 5 | Count data: log link and Poisson NLL `sec:poisson` | 1.5 | `PoissonNLLLoss`; `poisson_regression.py` | First genuinely new pairing. log link, NLL = e^z - yz, gradient e^z - y (third theorem instance). Real `PoissonNLLLoss` listing. Worked experiment and honest findings (both heads fit a benign target; the structural argument is load-bearing; min-rate 0.068 vs 0.310). Numbers from the notebook. |
| 6 | Uncertainty: heteroscedastic Gaussian NLL `sec:heteroscedastic` | 1.75 | `GaussianNLLLoss`, `softplus`; `heteroscedastic.py` | Two-output head; mu = z_mu, sigma = softplus(z_s); NLL; hand-derived gradients through softplus. Real `GaussianNLLLoss` listing. The prediction table and the test NLL 0.745 vs 0.509 (0.24 nat calibration premium). Numbers from the notebook. |
| 7 | Link and likelihood are independent choices `sec:link-likelihood` | 1.0 | booklet ch2 §7 | Same link (sigmoid), two likelihoods (Gaussian-on-output vs Beta). Breaks the catalogue's apparent one-to-one. Beta NLL is a natural extension (needs only `math.lgamma`), not a library gap. |
| 8 | Beyond unimodal: a pointer to mixture density networks `sec:mdn` | 0.5 | booklet ch2 §8 | Multimodal p(y given x); the MDN head; composes softplus, softmax, logsumexp. Natural extension, framed exactly like the autograd note in `\Cref{sec:autograd-note}`. |
| 9 | Inductive bias, the parallel axis `sec:parallel-axes` | 1.0 | booklet ch2 §9 | Owns the full two-axes synthesis: head and architecture are independent and composable; the matched-vs-mismatched analysis; the composability examples (CNN+Poisson, Transformer+Categorical, etc.). `\Cref{sec:inductive-biases}` previews; this section completes. |
| 10 | Library additions `sec:library-additions` | 0.5 | `scratchnn` | `softplus`, `MSELoss`, `PoissonNLLLoss`, `GaussianNLLLoss`, all fitting the `Loss` interface, each with a `gradient_check` case. Present tense: these are already in the library. Beta and MDN remain out, as natural extensions. |
| 11 | Handoff `sec:handoff` | 0.5 | booklet ch2 §11 | Recap, then forward. FIX the ordering error: the architecture axis continues in the NEXT chapter, CNN (`\Cref{ch:cnn}`), not "already done". Heads compose with any body; RL closes the book. |

### Chapter 2 Notebook (`notebooks/ch02-output-heads.ipynb`)

- Poisson experiment: lambda(x) = max(0.1, 2 + 5 sin(pi x)) on x in [0, 2]; identical 1-16-1 Tanh bodies, MSE vs Poisson heads. Regenerate the min-predicted-rate values (0.068 vs 0.310) and the comparative Poisson NLL.
- Heteroscedastic experiment: y = sin(x) + noise with sigma(x) = |x|/3 + 0.1 on [0, 6]; 1-16-1 (MSE) vs 1-16-2 (Gaussian NLL) bodies. Regenerate the six-row prediction table and the test NLL comparison (0.745 vs 0.509).
- `gradient_check` cases for `MSELoss`, `PoissonNLLLoss`, `GaussianNLLLoss` (anchor on a Tanh MLP; residuals near 1e-10).
- Seeds recorded in the notebook header so the prose can be written to the regenerated values per decision (C).
- Uses NumPy where convenient for the experiments; the loss classes themselves are pure-Python `scratchnn`.

## 7. Forward and Backward Reference Map (reconciled to book order)

| From | Direction | To | Note |
|---|---|---|---|
| `sec:nn-as-function` | forward | `sec:logistic`, `sec:softmax` | the residual `p - y` first seen here, derived there |
| `sec:mlp-backprop` | forward | all of Parts II and III | the `Layer` protocol and backprop reused unchanged |
| `sec:softmax` | forward (intra-chapter) | `sec:inductive-biases` | "why is the digit gap so small?" answered by the bias argument |
| `sec:logits-logprobs` | forward (intra-chapter) | `sec:stability` | the additive-constant gauge reused for max-subtraction |
| `sec:inductive-biases` | forward | `ch:output-heads`, Part II | output-head axis to Ch 2; architecture axis to Part II. Replaces "follow-up post" |
| `sec:autograd-note` | forward | `sec:mdn`, `sec:link-likelihood` | the "natural extension, not built" template |
| `sec:mle-frame` | backward | `ch:foundations` | "logits interpreted by the loss" is this MLE frame |
| `sec:canonical-link` | backward | `sec:logistic`, `sec:softmax` | proves the `p - y` "no coincidence" |
| `sec:canonical-link` | forward (intra-chapter) | `sec:poisson` | third instance of the theorem |
| `sec:three-pairings` | backward | `ch:foundations` | the three pairings Ch 1 built |
| `sec:parallel-axes` | backward | `sec:inductive-biases` | completes the synthesis Ch 1 previewed |
| `sec:handoff` | forward | `ch:cnn` (Ch 3) | FIX: CNN is the next chapter, not prior |

The single most important fix: in book order, Chapter 2 (output heads) precedes Chapter 3 (CNN). The booklet text calls the CNN chapter "already done" and treats it as prior work. Every such reference is reconciled to a forward `\Cref{ch:cnn}`.

## 8. Page Budget

| Chapter | Pages | Notes |
|---|---|---|
| Part intro (`\part{Foundations}` blurb) | ~1 | port the booklet part-1 blurb |
| Ch 1 Foundations | ~11 | code-heavy; overflow worked-output to the notebook |
| Ch 2 Output Heads | ~10 | results tables kept minimal in prose, rest in the notebook |
| **Total** | **~22** | matches the master spec Part I budget |

If a chapter overruns, the first lever is to move a worked numerical block into the paired notebook, not to expand prose. Code listings stay trimmed (init elided, `backward` kept in full).

## 9. Sequencing for the Implementation Plans

Draft in chapter order. Chapter 1 first: it establishes the `Layer` protocol, the scratchnn core, the `p - y` seed, and the two-axes thesis that Chapter 2 builds on. Then Chapter 2.

For each chapter, the plan runs notebook-first (decision C):

1. Port or write the paired notebook, fix seeds, execute end to end, and record the actual numbers.
2. Write the chapter plan (`/bookwright:plan chapterN`) with the verified numbers and the section table above.
3. Draft the prose to the notebook's output; pull every code listing from `src/scratchnn` and wire `\label`s.
4. Apply the de-serialization pass (decision B) as an explicit plan checklist item.
5. Audit (spec, quality, math, cross-reference) and integrate.

Chapter 1 plan must verify these listings against source: the `Layer` base, `Linear`, `Tanh`, `ReLU`, `Network.forward`/`backward`, `zero_grad`/`step`, `SigmoidBCE`, `SoftmaxCrossEntropy`, the `sigmoid`/`logsumexp`/`softmax` helpers, and `gradient_check`. Chapter 2 plan must verify `PoissonNLLLoss`, `GaussianNLLLoss`, and `softplus`, and regenerate the Poisson and heteroscedastic numbers.

## 10. Open Action Items

1. **Resolved: output-heads code gap.** The library implements all five heads (Section 4). The master spec has been patched. No action beyond keeping listings in sync at integration.
2. **Code-link treatment in print.** The booklet uses `\href` to GitHub for `examples/*.py` and `tests/*.py`. Decide the book form: a single front-matter or colophon statement that all code lives in the repo at a canonical URL, plus repo-relative `\texttt{}` paths inline (preferred), rather than a live hyperlink per listing. Resolve in the Chapter 1 plan and apply book-wide.
3. **Notebook seeds.** Choose and record seeds for the XOR, digits, Poisson, and heteroscedastic runs so the numbers are reproducible. The chapter plans pin these before prose is written.
4. **Overlap boundary.** Pin exactly what `\Cref{sec:inductive-biases}` keeps (name the two axes, the digits payoff, the one-number CNN-vs-MLP teaser) versus what moves wholly to `\Cref{sec:parallel-axes}` (the composability matrix, the matched/mismatched analysis). Decide in the Chapter 1 and Chapter 2 plans together.
5. **Pairings count language.** Chapter 1 commits to three head examples; Chapter 2 extends to Poisson, heteroscedastic Gaussian, and Beta. Keep the count language consistent across both so the reader is not told "three" and then "five" without the extension being explicit.
6. **`\part` blurb and front matter.** Port the booklet's one-paragraph Part I blurb into `book/parts/part1.tex` (currently a bare `\part{Foundations}` stub) and confirm the preface and notation front matter port cleanly with the no-em-dash rule.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Auto-numbering conversion (decision A) churns labels and breaks references mid-draft. | The label scheme is fixed in Section 4 of this spec. Wire `\label` at section creation; the cross-reference auditor verifies `\Cref` resolution at integration. |
| Trimming the `sec:inductive-biases` / `sec:parallel-axes` overlap removes the param-count payoff readers liked. | Keep the one-number CNN-vs-MLP teaser in Chapter 1; the full composability analysis lives in Chapter 2. Action item 4 pins the boundary. |
| Re-running the notebooks shifts Chapter 2's specific numbers, forcing late prose edits. | Notebook-first sequencing (Section 9). Prose is written to the regenerated values, never the reverse. |
| Chapter 1 is long (10 sections, six code listings) and overruns ~11 pp. | Listings trimmed (init elided, `backward` in full); worked output to the notebook; the autograd note stays at half a page. |
| The de-serialization pass misses a blog-order reference and a forward chapter is described as prior. | The reference map (Section 7) is the checklist; the spec-auditor checks it against the drafted text. |

## 12. Out of Scope (Part I)

- General GLM machinery beyond the single canonical-link theorem (`\Cref{sec:canonical-link}` proves exactly what is needed: no Fisher scoring, no IRLS).
- Optimization theory: SGD is plain mini-batch, no convergence analysis.
- Beta NLL and MDN implementations: natural-extension pointers only, not built.
- Any Part II or Part III content (convolution internals, attention, interpretability, RL) beyond the forward-pointing teasers in `\Cref{sec:inductive-biases}`, `\Cref{sec:parallel-axes}`, and `\Cref{sec:handoff}`.

## 13. Success Criteria for Part I

1. A self-study reader finishing Part I can build an MLP from the three-method `Layer` protocol, explain why a single linear unit cannot fit XOR, state the `p - y` gradient and why it recurs (the canonical-link theorem), and select and justify an output head for real-valued, binary, categorical, count, and heteroscedastic targets.
2. Every code listing matches current `src/scratchnn`, verified at integration.
3. Both chapters' numbers regenerate from their paired notebooks with recorded seeds.
4. No blog-serialization artifacts remain: Chapter 2 forward-references CNN (`\Cref{ch:cnn}`), no "post" or "follow-up post" idiom survives, and the two-axes synthesis lives in Chapter 2 with only a preview in Chapter 1.
5. `\Cref` resolves for every cross-reference; no manual section numbers appear in titles.
6. The `p - y` thread is visibly carried: introduced in `\Cref{sec:logistic}`, generalized in `\Cref{sec:canonical-link}`, instanced again in `\Cref{sec:poisson}`.

## 14. Next Step

`/bookwright:plan chapter1` to write the Chapter 1 implementation plan (notebook-first, per Section 9). Note that the plan should also pin the cross-reference label `\label{ch:cnn}` expectation for Chapter 3 so Chapter 2's forward reference resolves once Part II exists.
