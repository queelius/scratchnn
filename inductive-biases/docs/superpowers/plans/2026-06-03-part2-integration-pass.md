# Integration-Pass Record: Part II (Architectural Inductive Biases)

**Date:** 2026-06-03
**Scope:** Part II, Chapters 3 (CNN), 4 (Fixed-Context LM), 5 (RNN), 6 (Transformer)
**Plans:** `docs/superpowers/plans/2026-06-03-chapter{3,4,5,6}-*.md`
**Specs:** `2026-06-02-master-design.md`, `2026-06-03-part2-design.md`
**Review:** `docs/superpowers/reviews/2026-06-03-part2.md` (full multi-agent part-level review; 0 blocking, 2 substantive, 6 minor; S1 and S2 fixed, see below)
**HEAD at integration:** `3989dd8` (branch `inductive-biases-part2`)
**Verdict:** PASS. Part II is integrated and sealed; ready to merge to `main`. Only Part III (Chapters 7, 8) remains.

## 1. Verification results

| Check | Result | Detail |
|---|---|---|
| Build (`make cleanall && make`) | PASS | exit 0 |
| Undefined references | PASS | 0 |
| Overfull hboxes (heads / total) | PASS | 0 / 0 |
| Undefined control sequences | PASS | 0 |
| Label uniqueness (ch3-6) | PASS | 50 labels, 0 collisions |
| Soul voice (em-dashes) | PASS | 0 in ch3-6 |
| Blog idiom ("post") | PASS | 0 |
| Macro / pandoc leaks | PASS | 0 |
| Notebooks executed with stored outputs | PASS | ch3 8/8, ch4 7/8, ch5 8/11, ch6 12/12 (the non-output cells are imports/defs) |
| Page budget | PASS (within tolerance) | ~40 pp vs ~37 pp Part II target |

## 2. Cross-reference map

**Defined in Part II (50 labels):** `ch:cnn`, `ch:fixed-context-lm`, `ch:rnn`, `ch:transformer`, plus per-chapter `sec:*`, `eq:*`, `fig:*`, and `tab:depth-transition`.

**Backward `\Cref` into Part I (resolve):** `ch:foundations`, `sec:mlp-backprop`, `sec:inductive-biases`, `sec:parallel-axes`, `sec:softmax`. Within Part II: `ch:cnn` referenced by ch5 (weight sharing transfers) and ch4 (1-D CNN map row); `ch:fixed-context-lm` by ch5 (the absorbed Bengio comparison); all four chapters by ch6's recap and synthesis. All resolve.

**Expected-unresolved baseline: EMPTY.** Forward references to Chapter 7 (interpretability) and Chapter 8 (RL) are prose-only by the cross-reference discipline; the build shows zero undefined references.

## 3. Running-thread inventory (verdicts from the part-level review)

| Thread | Status |
|---|---|
| Architecture as a prior over feature composition; per-chapter "state the prior, build the smallest model, test it" | PASS. CNN locality/equivariance (permuted-pixel control), Bengio bounded window, RNN time-equivariance/bottleneck, Transformer content-addressable lookup. |
| Decision-A seam (Ch4 previews the RNN forward; Ch5 absorbs the full Bengio comparison) | PASS. No "the RNN in the previous chapter" in Ch4; `\Cref{ch:fixed-context-lm}` resolves from Ch5; no duplication. |
| The third axis (implementation realization) revealed only in Ch6 | PASS. Two axes through Ch3-5; the third emerges from Ch6's scaling investigation, not retrofitted. |
| `scratchnn` through-line giving way to NumPy at Ch6 | PASS. Pure-Python Conv2D/Embedding/EmbedConcat/RNNCell (Ch3-5); the NumPy switch is stated plainly at Ch6; the PyTorch M=32 sweep is cited, not run. |
| Real-code fidelity per chapter | PASS. Listings match `conv.py`, `embedding.py`, `recurrent.py`, `examples/transformer.py`. |

## 4. Page budget

Part II divider pages 29 to 32; Ch3 pages 33 to 40 (~8 pp); Ch4 pages 41 to 48 (~8 pp); Ch5 pages 49 to 58 (~10 pp); Ch6 pages 59 to 72 (~14 pp); Bibliography begins page 73. Part II chapter content is ~40 pp against the ~37 pp spec target, within tolerance (Ch5's absorbed comparison and Ch6's investigation density account for the overage).

## 5. Substantive review findings, resolved

- **S1 (Ch6 scaling iteration count).** The scaling result was internally inconsistent (table "6000 it", figure/notebook "3000"). Reconciled to the authoritative `examples/RESULTS.md` config: 2 layers, 1 head, d_model 64, sinusoidal PE, 6000 iters (the 0.747 / 0.687 / 0.664 plateau points). Figure, caption, notebook, and the summary print now all read 6000 iters / d=64. Commits `4bbdc1b`, `3989dd8`.
- **S2 (Ch6 scaling figure provenance).** The caption now states the curve is plotted from the recorded runs in `examples/RESULTS.md` (the grokking figure remains a live re-run). Commit `4bbdc1b`.

## 6. Ch6 notebook receipt note

The Ch6 notebook is NumPy-only and executed end to end (12/12 code cells carry stored outputs). The grokking experiment is a genuine live re-run (figure `ch06-grokking.pdf`; MLP 1.000, 1L 0.654, 2L groks to 1.000, transition near iter 800). The M-scaling figure (`ch06-scaling.pdf`) plots the recorded `RESULTS.md` data because the full sweep is a ~30-minute pure-NumPy run; `run_sweep` is kept in the notebook to reproduce it live. The PyTorch M=32 depth result is cited as a table with a non-comparability caveat, never run. This is the Part II spec's sanctioned approach for the one expensive chapter.

## 7. Open follow-ups (non-blocking; for a future polish pass)

From the review's minor findings, none blocking:
- M1: Ch5 bib-notes header block omits `goodfellow2016dlbook` (cited in prose; resolves fine). Header-comment hygiene.
- M2: the Ch3 plan/spec carry a stale "1.51" permuted-pixel estimate; the chapter correctly tracks the notebook's 1.50. Spec erratum.
- M4: the Ch5 notebook has a markdown cell saying 2.06 while the chapter (correctly) says 2.05. Notebook-only.
- M3, M5, M6: trivial or within-policy (a sample-seed string length, lightly-renamed conv locals, uneven table budget headers now mostly resolved by S1).

## 8. Verdict

Part II (Architectural Inductive Biases) is integrated and sealed. The build is clean with an empty unresolved-reference baseline, all four notebooks are executed, the four running threads carry coherently across the four chapters, the decision-A order fix and the third-axis reveal both hold, and every code listing matches source. Part II merges to `main`. The book is now six of eight chapters complete; Part III (Chapter 7 interpretability, Chapter 8 reinforcement learning) remains.
