# Implementation Plan: Chapter 3, Convolutional Networks

**Author:** Alexander Towell
**Date:** 2026-06-03
**Status:** Plan (ready for `/bookwright:draft chapter3`)
**Specs:** `docs/superpowers/specs/2026-06-02-master-design.md`, `docs/superpowers/specs/2026-06-03-part2-design.md`
**Base SHA:** `fecc345` (branch `inductive-biases-part2`, with Part I merged)
**Scope:** Chapter 3 only (opens Part II), ~8 pp plus one paired notebook.

## 1. Goal

Draft Chapter 3, "Convolutional Networks: Locality and Translation Equivariance," by porting and elevating the booklet chapter. The chapter opens Part II (the architecture axis): a convolution is one linear unit reused at every position, weight sharing IS translation equivariance, and the parameter count tells the story. It shows the real pure-Python `Conv2D` forward and backward, trains a CNN on 8x8 digits against the Chapter 1 MLP, and runs the permuted-pixel control as a falsification test of the locality prior. It closes by marking the boundary: `Conv2D` is the last large layer written in pure Python, and the Transformer chapter will reach for NumPy.

## 2. Architecture Summary

- Chapter 3 is the first chapter of Part II, so its scaffold creates `book/parts/part2.tex` (the `\part{Architectural Inductive Biases}` divider and blurb, ported from the booklet) and adds `\input{parts/part2.tex}` to `book.tex` after Part I. `part2.tex` then `\input`s the Chapter 3 wrapper.
- Per-file structure under `book/chapters/ch03/`, one file per drafting task, each with a cross-reference header block.
- Sections auto-numbered; `\Cref` for resolvable targets; prose-only forward refs; `listings` for code (already in the preamble). No new preamble additions (no theorem environment needed).
- Pure-Python `scratchnn`: `Conv2D`, `ReLU`, `Linear`, `Network`, `SoftmaxCrossEntropy`.

## 3. Tech Stack

LaTeX (book, pdflatex + biber, listings). Notebook: Python + uv + Jupyter, `scratchnn` editable, `scikit-learn` for `load_digits`, `matplotlib` for the permuted-pixel figure. Pure-Python `scratchnn` core (no NumPy in the model).

## 4. File Structure

```
book/
  book.tex                                 (Task 1: + \input{parts/part2.tex})
  parts/part2.tex                          (Task 1: \part divider + blurb + \input ch03)
  chapters/ch03/
    ch03.tex                               (Task 1: \chapter + opening + \inputs)
    01-conv-and-sharing.tex                (Task 3: sec 1-3)
    02-conv-code.tex                       (Task 4: sec 4-5, the Conv2D listings)
    03-digits-and-control.tex              (Task 5: sec 6-7 + figure)
    04-bias-and-boundary.tex               (Task 6: sec 8-9)
    bib-notes.tex                          (Task 7)
  figures/ch03-permuted-pixel.pdf          (Task 2: notebook output)
notebooks/ch03-cnn.ipynb                   (Task 2)
```

No exercises file. Bibliographic Notes are an unnumbered `\section*`.

## 5. Cross-reference Map

**Labels DEFINED in Chapter 3:**

| Label | File | Kind |
|---|---|---|
| `ch:cnn` | ch03.tex | chapter (resolves Chapter 2's prose-only CNN forward ref) |
| `sec:conv-unit` | 01-conv-and-sharing | section |
| `sec:weight-sharing` | 01-conv-and-sharing | section |
| `sec:conv-params` | 01-conv-and-sharing | section |
| `sec:conv-forward` | 02-conv-code | section |
| `sec:conv-backward` | 02-conv-code | section |
| `eq:conv-grad` | 02-conv-code | the conv kernel gradient |
| `sec:conv-digits` | 03-digits-and-control | section |
| `sec:permuted-pixel` | 03-digits-and-control | section |
| `fig:permuted-pixel` | 03-digits-and-control | figure |
| `sec:cnn-bias` | 04-bias-and-boundary | section |
| `sec:pure-python-boundary` | 04-bias-and-boundary | section |

**Resolvable backward `\Cref` into merged Part I (all resolve):** `\Cref{ch:foundations}`, `\Cref{sec:mlp-backprop}` (the per-layer backward and `+=` accumulation), `\Cref{sec:inductive-biases}` (the two axes; `sec:cnn-bias` returns to it).

**Forward references (PROSE-ONLY, not `\Cref`):** Chapter 4 (sequences / the fixed-context LM, "the next chapter"), Chapter 6 (the Transformer, where pure Python ends). Both prose-only since those chapters are not yet drafted.

**Expected-unresolved baseline for Chapter 3: EMPTY.** Backward refs resolve; forward refs prose-only. A clean `make` must show zero "Reference undefined."

**Naming note (for a later chapter's backward ref):** Chapter 5 (RNN) will refer back to this chapter's falsification experiment. Name it consistently the **permuted-pixel control** (the Part II spec flags a stale "shapes experiment" label in the Ch5 source to fix against this name).

## 6. Lessons Inherited (from Parts I plans and reviews, and the Part II spec)

- **Notebook-first with stored outputs.** Run `ch03-cnn.ipynb`, fix seeds, execute end to end so all cells carry outputs, then write prose to the regenerated numbers. (Part I lesson: a notebook without stored outputs is not a receipt; a non-executing notebook gave uncertified numbers in Ch2.)
- **Place figures, do not just generate them** (Ch1 finding S1). The notebook saves `figures/ch03-permuted-pixel.pdf`; `sec:permuted-pixel` must `\includegraphics` it with a `\Cref`; integration verifies the caption is in the PDF.
- **Regime-change reconciliation** (Ch1 digits, Ch2 Poisson). If a fresh seeded run shifts an accuracy or a parameter count, update every sentence that interprets it. The permuted-pixel gap (CNN drop > MLP drop) is the load-bearing claim; confirm the direction and magnitude survive the re-run.
- **Real code only**, from `../../src/scratchnn/conv.py` and the Part I core, lightly trimmed (backward kept in full).
- **Clean-rebuild to verify** (`make cleanall && make`); the Makefile bug is fixed but the habit stays. Distinguish overfull `\hbox` (real) from benign underfull `\vbox`.
- **Voice**: no em-dashes (the soul hook blocks writes); avoid LLM filler; the soul list is authoritative.
- **Cross-ref discipline**: per-file header blocks; `\Cref` for resolvable, prose-only for forward chapters; short chapter title (`\chapter[Convolutional Networks]`) to avoid running-head overflow.
- **Two axes only** (Part II spec decision D): Chapter 3 speaks of the output-head and architecture axes; it may forward-hint that realization is coming but does not name a third axis (that is Chapter 6).
- **De-serialize from the booklet `.tex`** (about 90 percent clean), never the series `.md` (full of "post" idiom).

## 7. Source Material

| Source | Use |
|---|---|
| `../../docs/booklet/chapters/03-cnn.tex` | primary port source |
| `../../docs/series/03-cnn.md` | prose seed and voice exemplar |
| `../../src/scratchnn/conv.py` | real code: `Conv2D` (forward five-loop, hand-derived backward), `GlobalAvgPool` |
| `../../examples/digits_cnn.py`, `../../examples/digits.py` | notebook: the CNN-vs-MLP digits experiment and the permuted-pixel control |
| `../../docs/booklet/parts/part2.tex` | the Part II divider blurb to port into `book/parts/part2.tex` |
| merged Part I (`book/chapters/ch01/`) | backward `\Cref` targets (`ch:foundations`, `sec:mlp-backprop`, `sec:inductive-biases`) |

## 8. Tasks

Eight tasks. Notebook is Task 2 (notebook-first). No exercises task.

### Task 1: Scaffold (open Part II, chapter wrapper)

- Create `book/parts/part2.tex`: `\part{Architectural Inductive Biases}` plus the ported one-paragraph Part II blurb (from the booklet `parts/part2.tex`), then `\input{chapters/ch03/ch03}`.
- In `book/book.tex`: add `\input{parts/part2.tex}` immediately after `\input{parts/part1.tex}` in the mainmatter.
- Create `book/chapters/ch03/ch03.tex`: chapter header block; `\chapter[Convolutional Networks]{Convolutional Networks: Locality and Translation Equivariance}\label{ch:cnn}`; the chapter-opening prose (architecture as a prior over how features compose; this opens Part II); then `\input` the four section files and `bib-notes`.
- Create placeholder section files (header block + `\section{...}\label{...}` + `% TODO Task N`) so the chapter compiles.
- **Verify:** `make -C book` compiles; TOC shows Part II and Chapter 3; zero undefined refs.
- Commit: `ch03: scaffold Part II, chapter wrapper, part wiring`.

### Task 2: Paired notebook (run first, lock numbers and the figure)

`notebooks/ch03-cnn.ipynb`. Port from `examples/digits_cnn.py` and `examples/digits.py`.

- UCI optdigits (3823 train / 1797 test, pixels normalized to [0,1]). Train the MLP baseline `64-32-10` and the CNN `Conv2D(1,4,k=3) -> ReLU -> Linear(144,10)` (40 epochs, lr 0.1, batch 32). Record test accuracies (target MLP ~96.1%, CNN ~95.4%) and parameter counts (MLP 2410, CNN 1490 = 40 conv + 1450 head; conv 40 vs `Linear(64,144)` 9360 = 234x).
- Permuted-pixel control: apply one fixed random permutation to the 64 pixels for both models, retrain, record the accuracy drop (target MLP -0.45pp, CNN -1.51pp; CNN drop > 3x MLP drop is the falsification signature).
- `gradient_check` on a small `Conv2D` (target ~1e-9).
- Fix and record seeds; execute end to end with stored outputs; a Results cell with the exact numbers.
- **Figure:** `book/figures/ch03-permuted-pixel.pdf` (standard vs permuted accuracy, CNN vs MLP, e.g. grouped bars).
- Commit: `ch03: paired notebook, seeded and executed; lock numbers and figure`.

### Task 3: Sections 1 to 3, the conv unit, weight sharing, parameter count

File `01-conv-and-sharing.tex`. ~2.25 pp. Header: DEFINED `sec:conv-unit`, `sec:weight-sharing`, `sec:conv-params`; RESOLVED `\Cref{ch:foundations}`, `\Cref{sec:mlp-backprop}`.

- Checklist: a convolution is one linear unit (a k x k kernel) slid across the input; the window is the locality prior. Weight sharing (one kernel reused at every position) IS translation equivariance: a detector that fires at one location fires anywhere. The parameter count: a `Conv2D(1,4,k=3)` has 40 parameters versus an equivalent `Linear(64,144)` with 9360, a 234x reduction (label `eq:conv-params` if a display is used); forward cost 1296 multiply-adds per example. The prior in one number.
- Commit: `ch03: sections 1-3 the conv unit, weight sharing, parameter count`.

### Task 4: Sections 4 to 5, the Conv2D forward and backward, in code

File `02-conv-code.tex`. ~2 pp. Header: DEFINED `sec:conv-forward`, `sec:conv-backward`, `eq:conv-grad`; RESOLVED `\Cref{sec:mlp-backprop}`.

- Checklist: the real `Conv2D.forward` listing (the five-nested-loop form), described as the largest layer written in pure Python. The real `Conv2D.backward` listing: the hand-derived kernel gradient with `+=` accumulation across positions (label `eq:conv-grad`), tying back to the per-layer backward protocol of `\Cref{sec:mlp-backprop}`. Note the same `Layer` contract from Chapter 1 carries unchanged.
- Commit: `ch03: sections 4-5 the Conv2D forward and backward`.

### Task 5: Sections 6 to 7, training on digits and the permuted-pixel control

File `03-digits-and-control.tex`. ~2 pp. Header: DEFINED `sec:conv-digits`, `sec:permuted-pixel`, `fig:permuted-pixel`; RESOLVED (none new).

- Checklist (sec 6): CNN 1490 params reaching ~95.4% vs the Chapter 1 MLP 2410 params at ~96.1%: smaller and competitive. Note the 8x8 digits are pre-pooled from 32x32 NIST by 4x4 block-counting (already a local feature extractor), explaining the small absolute gap. Numbers from the notebook.
- Checklist (sec 7): the **permuted-pixel control** as the falsification test. A fixed random permutation of the 64 pixels destroys adjacency; if the CNN truly uses locality it should lose more than the MLP. Result: MLP -0.45pp, CNN -1.51pp (CNN drop > 3x). **Place `fig:permuted-pixel`** with `\includegraphics` and a `\Cref`. Numbers from the notebook.
- Commit: `ch03: sections 6-7 digits training and the permuted-pixel control`.

### Task 6: Sections 8 to 9, inductive bias named and the pure-Python boundary

File `04-bias-and-boundary.tex`. ~1.25 pp. Header: DEFINED `sec:cnn-bias`, `sec:pure-python-boundary`; RESOLVED `\Cref{sec:inductive-biases}`.

- Checklist (sec 8): locality plus translation equivariance as a bet; the right prior is empirical (a CNN beats an MLP on images because the prior matches; it loses on tabular data where the spatial prior is a lie). Return to the two axes via `\Cref{sec:inductive-biases}`. Two axes only (no third axis here).
- Checklist (sec 9): `Conv2D` is the last large layer written in pure Python; forward-promise (prose) that the Transformer chapter reaches for NumPy. One-line natural-extension pointer to `GlobalAvgPool` (collapsing an equivariant stack to an invariant classifier), in the manner of Chapter 1's autograd note.
- Commit: `ch03: sections 8-9 inductive bias named and the pure-Python boundary`.

### Task 7: Bibliographic Notes

File `bib-notes.tex`, unnumbered `\section*{Bibliographic Notes}`. ~0.5 pp. Header: RESOLVED `\cite` keys only.

- Checklist: Fukushima (1980, neocognitron); LeCun et al. (1989, backpropagation applied to handwritten zip codes) and LeCun et al. (1998, gradient-based learning / LeNet); the UCI optdigits dataset (reuse `uci1998digits`); optionally Goodfellow et al. (convolutional networks chapter). Add new entries (`fukushima1980`, `lecun1989`, `lecun1998`) to `book/references.bib`; confirm each `\cite` resolves under biber.
- Commit: `ch03: bibliographic notes and references.bib entries`.

### Task 8: Integration and verification

- **Build:** `make -C book cleanall && make -C book` clean, exit 0.
- **Cross-reference audit:** every `\label` defined; backward `\Cref`s into Part I resolve; `book.log` shows zero "Reference undefined" (empty baseline). Optionally upgrade Chapter 2's prose-only CNN forward reference to `\Cref{ch:cnn}` now that it exists (low priority).
- **Figure placed (S1):** confirm `fig:permuted-pixel` is `\includegraphics`'d and its caption appears in the PDF.
- **Real-code audit:** diff the `Conv2D` forward/backward listings against `conv.py`.
- **Numbers audit:** accuracies, parameter counts (2410, 1490, 40, 9360, 234x), and the permuted-pixel drops reconciled with the notebook.
- **De-serialization audit:** no "post" idiom; the experiment is named "permuted-pixel control"; CNN forward-references (not "already done") to Chapters 4 and 6.
- **Voice / heads:** no em-dashes; 0 overfull hboxes in heads.
- **Page budget:** ~8 pp.
- Commit: `ch03: integration pass; build clean, figure placed, listings verified`.

## 9. Risks

| Risk | Mitigation |
|---|---|
| The permuted-pixel gap shrinks or inverts on a fresh seed, weakening the falsification claim. | Notebook-first: confirm CNN drop > MLP drop survives the re-run; if the magnitude shifts, update the prose to the actual drops (the claim is the direction and the >2x ratio, not exact pp). |
| A pure-Python CNN training run (40 epochs on 3823 examples) is slow in the notebook. | Keep the 4-channel, k=3 model small as specified; if runtime is a problem, reduce epochs and update numbers, or cache, but keep both models on equal footing. |
| `examples/digits_cnn.py` differs from the planned config. | Use the example as the basis; reconcile the reported numbers to whatever the seeded notebook actually produces. |
| Conv backward listing is long. | Trim init/boilerplate; keep the hand-derived gradient and the `+=` accumulation in full. |

## 10. Out of Scope (Chapter 3)

- Pooling beyond the one-line `GlobalAvgPool` pointer; no max-pool, no multi-layer conv stacks.
- Padding, stride, dilation (the library `Conv2D` is no-padding, stride 1).
- NumPy or PyTorch (Chapter 3 is pure Python; the switch is forward-promised, not taken).
- Exercises (book has none).

## 11. Success Criteria for Chapter 3

1. A reader can explain why weight sharing is translation equivariance, read the parameter-count argument, and understand the permuted-pixel control as a falsification test of the locality prior.
2. The `Conv2D` listings match `conv.py`; the digits and permuted-pixel numbers regenerate from the notebook; `fig:permuted-pixel` is placed.
3. Backward `\Cref`s into Part I resolve; forward references to Chapters 4 and 6 are prose-only; zero undefined refs.
4. Part II is opened correctly: `part2.tex` exists with the blurb, `book.tex` inputs it, `ch:cnn` is defined.
5. Two axes only; the pure-Python boundary is marked and the NumPy switch forward-promised.

## 12. Next Step

`/bookwright:draft chapter3` to execute this plan (scaffold and notebook first, then prose, then integration). After Chapter 3, the Part II sequence continues with `/bookwright:plan chapter4`.
