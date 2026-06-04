# Integration-Pass Record: Part I (Foundations)

**Date:** 2026-06-03
**Scope:** Part I, Chapters 1 (Foundations) and 2 (Output Heads as Inductive Bias)
**Plans:** `docs/superpowers/plans/2026-06-03-chapter1-foundations.md`, `2026-06-03-chapter2-output-heads.md`
**Specs:** `docs/superpowers/specs/2026-06-02-master-design.md`, `2026-06-02-part1-design.md`
**Reviews:** per-chapter `2026-06-03-ch01.md`, `2026-06-03-ch02.md`; part-level `2026-06-03-part1.md`
**HEAD at integration:** `e12da73` (branch `inductive-biases-part1-review`)
**Verdict:** PASS. Part I is integrated and sealed; Part II may be built on top of it.

## 1. Verification results

| Check | Result | Detail |
|---|---|---|
| Build (`make cleanall && make`) | PASS | pdflatex + biber + pdflatex x2, exit 0 |
| Undefined references (`book.log`) | PASS | 0 |
| Overfull hboxes in running heads | PASS | 0 |
| Overfull hboxes total | PASS | 0 |
| Undefined control sequences | PASS | 0 |
| Label uniqueness (ch1+ch2) | PASS | 34 labels, 0 collisions |
| Cross-reference resolution | PASS | 23 distinct `\Cref` targets, all resolve |
| Soul voice (em-dashes) | PASS | 0 in authored Part I files |
| Macro / pandoc leaks | PASS | no `hypertarget` / `tightlist` / `passthrough` |
| Notebooks executed with stored outputs | PASS | ch1 (16 cells) and ch2 (15 cells), all code cells carry outputs |
| Page budget | PASS (within tolerance) | ch1 ~12 pp, ch2 ~12 pp, ~24 pp vs ~22 pp target |

## 2. Cross-reference map

**Defined in Chapter 1:** `ch:foundations`; `sec:nn-as-function`, `sec:linear-limits`, `sec:mlp-backprop`, `sec:logistic`, `sec:softmax`, `sec:logits-logprobs`, `sec:inductive-biases`, `sec:stability`, `sec:gradcheck`, `sec:autograd-note`; `eq:residual`, `eq:linear-grads`, `eq:py-binary`, `eq:py-softmax`, `eq:logodds`; `fig:xor-boundary`.

**Defined in Chapter 2:** `ch:output-heads`; `sec:mle-frame`, `sec:catalogue`, `sec:canonical-link`, `sec:three-pairings`, `sec:poisson`, `sec:heteroscedastic`, `sec:link-likelihood`, `sec:mdn`, `sec:parallel-axes`, `sec:library-additions`, `sec:handoff`; `thm:canonical-link`, `eq:expfam-grad`; `fig:ch02-poisson`, `fig:ch02-heteroscedastic`; `tab:head-catalogue`.

**Cross-chapter backward references (resolve at build):** Chapter 2 to Chapter 1 via `ch:foundations`, `sec:logistic`, `sec:softmax`, `sec:inductive-biases`, `sec:autograd-note`, `sec:gradcheck`, `eq:residual`. All resolve.

**Expected-unresolved baseline: EMPTY.** Every forward reference into Part II and Part III is deliberately prose-only (the cross-reference discipline for not-yet-drafted chapters), so the build shows zero undefined references.

## 3. Running-thread inventory

| Thread | Chapter 1 | Chapter 2 | Status |
|---|---|---|---|
| Three axes (output head, architecture, implementation realization) | named and previewed in `sec:inductive-biases`, synthesis deliberately deferred | synthesis owned in `sec:parallel-axes` | Carries. The preview-to-synthesis handshake was explicitly verified in the part-level review: no gap, no redundant repetition. |
| The `p - y` gradient motif | introduced (`eq:py-binary`, `eq:py-softmax`) | generalized (`thm:canonical-link`, `eq:expfam-grad`), instanced again (`sec:poisson`) | Consistent and paid off. The residual glyph is now bridged: Chapter 1's `y-hat - y` (`eq:residual`) is named as the Gaussian-mean instance of `p-hat - y`. |
| `scratchnn` through-line (real inlined code) | `Layer`, `Linear`, `Tanh`, `ReLU`, `Network`, `SigmoidBCE`, `SoftmaxCrossEntropy`, helpers, `gradient_check` | `MSELoss`, `PoissonNLLLoss`, `GaussianNLLLoss`, `softplus` | All listings verified against `src/scratchnn/neural_net.py`. |
| Honest empiricism | the digits accuracy gap kept honest (capacity is not the bottleneck) | Poisson "both heads work here; the matched head is structural insurance" | Carries. |

## 4. Page budget

Part I divider pages 1 to 4; Chapter 1 pages 5 to 16 (~12 pp); Chapter 2 pages 17 to 28 (~12 pp); Bibliography begins page 29. Part I chapter content is ~24 pp against the ~22 pp spec target, within tolerance. The slight overage is driven by code listings, two experiment figures, the heteroscedastic results table, and the canonical-link theorem environment.

## 5. Known deferred items

None unresolved. The following forward references are intentionally prose-only and resolve when later parts are drafted:

- Chapter 2 handoff "the next chapter, on convolutional networks" to Chapter 3 (`ch:cnn`, defined in the Part II plan). May optionally be upgraded from prose to `\Cref{ch:cnn}` once Chapter 3 exists.
- "Part II" architecture-axis references (Chapters 3 to 6).
- "a closing chapter on reinforcement learning" to Chapter 8.

## 6. Open follow-ups (non-blocking; for a future polish pass)

- M4 (cosmetic): the Chapter 2 intro phrase "continuous proportions" sits near the catalogue's Gamma / positive-continuous row; could be smoothed.
- M5 (non-issue, left as is): the Chapter 1 wrapper header uses "CONTAINS" vocabulary while section files use "DEFINED"; both are accurate.
- M2 (resolved in the ch2 fix pass): the heteroscedastic sigma-recovery range is stated as 10 to 16 percent, verified against the executed notebook.

**Recurring lessons to carry into the Part II plans:** (1) always clean-rebuild to verify, since the scaffold Makefile bug that silently reused stale builds is fixed but the habit matters; (2) each chapter's paired notebook must execute end to end with stored outputs; (3) place generated figures with `\includegraphics` plus a `\Cref`, do not stop at saving them; (4) when an experiment's seed, epochs, or dataset changes, re-read every sentence that interprets the result; (5) backward `\Cref` into merged chapters resolves, forward references to undrafted chapters stay prose-only.

## 7. Verdict

Part I (Foundations) is integrated and sealed. The build is clean with an empty unresolved-reference baseline, both notebooks execute with stored outputs, all code listings match the library source, the running threads carry coherently across both chapters with the preview-to-synthesis handshake intact, and the voice is consistent and em-dash-free. Part II (Architectural Inductive Biases) may be built on top of this foundation.
