# Implementation Plan: Chapter 5, Recurrent Networks

**Author:** Alexander Towell
**Date:** 2026-06-03
**Status:** Plan (ready for `/bookwright:draft chapter5`)
**Specs:** `docs/superpowers/specs/2026-06-02-master-design.md`, `docs/superpowers/specs/2026-06-03-part2-design.md`
**Base SHA:** branch `inductive-biases-part2` (`cf59289`; Chapters 3 to 4 drafted)
**Scope:** Chapter 5 (third chapter of Part II), ~8 pp plus one paired notebook.

## 1. Goal

Draft Chapter 5, "Recurrent Networks: Time-Translation Equivariance." The RNN runs the same computation at every step and threads a hidden state that summarizes the past. The chapter states the prior, shows the real stateful `RNNCell`, derives backprop through time, earns the right to name vanishing gradients as the price of the prior, and trains a char-level model on Alice. Two roles unique to this chapter: it **absorbs the full Bengio-versus-RNN comparison** that Chapter 4 deferred (decision A), and it refers back to Chapter 3's falsification experiment by its correct name, **the permuted-pixel control** (the source has a stale "shapes experiment" label to fix).

## 2. Architecture Summary

Third chapter of Part II; `book/parts/part2.tex` gains `\input{chapters/ch05/ch05}`. Per-file structure under `book/chapters/ch05/`. Sections auto-numbered; `\Cref` resolvable / prose-only forward; `listings` for code. Pure-Python `scratchnn`: `RNNCell`, `Linear`, `Network`, `SoftmaxCrossEntropy`.

## 3. Tech Stack

LaTeX (book, pdflatex + biber, listings). Notebook: Python + uv + Jupyter, `scratchnn` editable, `matplotlib`. Pure-Python `scratchnn` (no NumPy).

## 4. File Structure

```
book/
  parts/part2.tex                          (Task 1: + \input{chapters/ch05/ch05})
  chapters/ch05/
    ch05.tex                               (Task 1: \chapter + opening + \inputs)
    01-prior-and-cell.tex                  (Task 3: sec 1-2)
    02-equivariance-bptt-vanishing.tex     (Task 4: sec 3-5, the mechanics)
    03-experiment-and-comparison.tex       (Task 5: sec 6 + figure + absorbed Bengio comparison)
    04-pattern-and-handoff.tex             (Task 6: sec 7-8)
    bib-notes.tex                          (Task 7)
  figures/ch05-rnn-loss.pdf                (Task 2: notebook output)
notebooks/ch05-rnn.ipynb                   (Task 2)
```

## 5. Cross-reference Map

**DEFINED:** `ch:rnn`; `sec:rnn-prior`, `sec:rnn-cell`, `sec:time-equivariance`, `sec:bptt`, `sec:vanishing-gradients`, `sec:rnn-alice`, `sec:rnn-pattern`, `sec:rnn-handoff`; `eq:rnn-jacobian`; `fig:ch05-rnn-loss`.

**Resolvable backward `\Cref` (all resolve; Chapters 3 to 4 drafted):**
- `\Cref{ch:foundations}`, `\Cref{sec:mlp-backprop}` (the `+=` accumulation and per-layer backward).
- `\Cref{ch:cnn}` and `\Cref{sec:permuted-pixel}` (`sec:time-equivariance` notes the weight-sharing argument transfers from space to time; refer to Chapter 3's experiment by its correct name, the permuted-pixel control).
- `\Cref{ch:fixed-context-lm}` (`sec:rnn-alice` runs the full Bengio-versus-RNN comparison).

**Forward references (PROSE-ONLY):** Chapter 6 (the Transformer, "the next chapter"). Not drafted yet.

**Expected-unresolved baseline: EMPTY.**

## 6. Lessons Inherited

All prior lessons: notebook-first with stored outputs; place the figure (S1); regime-reconciliation; real code from source; clean-rebuild; no em-dashes; per-file header blocks; `\Cref` resolvable / prose-only forward; short chapter title; two axes only; de-serialize from the booklet `.tex`. Chapter-5-specific: (a) **absorb the full Bengio comparison** in `sec:rnn-alice` (decision A); (b) **fix the stale "shapes experiment"** label to "the permuted-pixel control" wherever Chapter 3 is referenced.

## 7. Source Material

| Source | Use |
|---|---|
| `../../docs/booklet/chapters/05-rnn.tex` | primary port source |
| `../../docs/series/05-rnn.md` | voice exemplar (blog idiom; do not copy structure) |
| `../../src/scratchnn/recurrent.py` | real code: `RNNCell` (stateful protocol, hand-derived BPTT) |
| `../../examples/text_rnn.py` | notebook experiment |
| merged Part I + Chapters 3 to 4 | backward `\Cref` targets |

## 8. Tasks

Eight tasks. Notebook is Task 2. No exercises.

### Task 1: Scaffold
- `book/parts/part2.tex`: add `\input{chapters/ch05/ch05}` after Chapter 4.
- `book/chapters/ch05/ch05.tex`: header block; `\chapter[Recurrent Networks]{Recurrent Networks: Time-Translation Equivariance}\label{ch:rnn}`; opening prose (same computation at every step; a hidden state summarizes the past); `\input` the four section files and `bib-notes`.
- Placeholder section files.
- **Verify:** compiles; TOC shows Chapter 5; zero undefined refs.
- Commit: `ch05: scaffold chapter and part wiring`.

### Task 2: Paired notebook (run first)
`notebooks/ch05-rnn.ipynb`. Port from `examples/text_rnn.py`.
- Char-level Alice (same 30,000-char corpus, 75 vocab as Chapter 4). `RNNCell(75,64) -> Linear(64,75)`; BPTT seq length 32, lr 0.5 (against the mean per-timestep gradient), global grad-norm clip 5, 15 epochs. Regenerate the loss trajectory (4.32 to ~2.06), per-epoch samples, parameter count (~13,835). The vanishing-gradient demonstration (Jacobian-norm decay over unroll depth). `gradient_check` on the unrolled cell (T=3; ~1e-9, tol 1e-4).
- Fix seeds; execute end to end with stored outputs; Results cell.
- **Figure:** `book/figures/ch05-rnn-loss.pdf` (loss per epoch).
- Commit: `ch05: paired notebook, seeded and executed; lock numbers and figure`.

### Task 3: Sections 1 to 2, the prior and the cell
File `01-prior-and-cell.tex`. ~1.75 pp. Header: DEFINED `sec:rnn-prior`, `sec:rnn-cell`; RESOLVED `\Cref{ch:foundations}`.
- Checklist: the prior (same computation each step; hidden state summarizes the past; unbounded history in principle). The real `RNNCell` listing: `h_t = tanh(W_xh x + W_hh h_prev + b)`, the stateful `forward(x, state) -> (h, h)` / `backward(g, dstate_next) -> (dx, dstate_prev)` protocol extending the Chapter 1 `Layer` contract.
- Commit: `ch05: sections 1-2 the prior and the cell`.

### Task 4: Sections 3 to 5, equivariance, BPTT, vanishing gradients
File `02-equivariance-bptt-vanishing.tex`. ~2.75 pp. Header: DEFINED `sec:time-equivariance`, `sec:bptt`, `sec:vanishing-gradients`, `eq:rnn-jacobian`; RESOLVED `\Cref{ch:cnn}`, `\Cref{sec:permuted-pixel}`, `\Cref{sec:mlp-backprop}`.
- Checklist (sec 3): time-translation equivariance is weight sharing across time; the argument transfers from `\Cref{ch:cnn}` (space to time). When citing Chapter 3's experiment, name it **the permuted-pixel control** (fix the stale "shapes experiment" label).
- Checklist (sec 4): unrolling and BPTT; the `RNNCell` backward with `+=` accumulation across time, chained to `\Cref{sec:mlp-backprop}`; gradient clipping (norm 5) and the divide-by-T mean-gradient step.
- Checklist (sec 5): vanishing gradients as the price of the prior. The Jacobian product `dh_{t+1}/dh_t = diag(1 - h^2) W_hh` (label `eq:rnn-jacobian`); the spectral radius governs decay or explosion. This is qualitative falsification: the long-range drift in the samples is the prior showing its limit.
- Commit: `ch05: sections 3-5 equivariance, BPTT, vanishing gradients`.

### Task 5: Section 6, the experiment and the Bengio comparison (decision-A absorption)
File `03-experiment-and-comparison.tex`. ~1.75 pp. Header: DEFINED `sec:rnn-alice`, `fig:ch05-rnn-loss`; RESOLVED `\Cref{ch:fixed-context-lm}`.
- Checklist: the char-level Alice RNN run; loss 4.32 to ~2.06 over 15 epochs; samples quoted; ~13,835 params. **The full Bengio-versus-RNN comparison (decision A):** same corpus as `\Cref{ch:fixed-context-lm}`; the RNN reaches 2.06 over 15 epochs with BPTT while Bengio's fixed-context model reached 2.04 in about 4 epochs on independent windows (no BPTT). What each gives up and gains: the RNN's unbounded-history-in-principle versus the vanishing-gradient limit; Bengio's hard cutoff at N versus its training simplicity. This is the comparison Chapter 4 forward-hooked. **Place `fig:ch05-rnn-loss`** with `\includegraphics` and a `\Cref`. Numbers from the notebook.
- Commit: `ch05: section 6 the experiment and the Bengio comparison`.

### Task 6: Sections 7 to 8, the pattern repeats and the handoff
File `04-pattern-and-handoff.tex`. ~1.25 pp. Header: DEFINED `sec:rnn-pattern`, `sec:rnn-handoff`; RESOLVED `\Cref{ch:cnn}`, `\Cref{sec:mlp-backprop}`.
- Checklist (sec 7): weight-sharing-equals-equivariance across MLP, CNN, RNN (the three-architecture pattern table); the `+=` accumulation that the library has used since Chapter 1's `Linear`, chained `Linear` to `Conv2D` to `RNNCell`.
- Checklist (sec 8): recurrence squeezes the past through a state bottleneck; attention takes the other path. Forward-promise the Transformer (the next chapter, prose-only). LSTM/GRU as a one-line natural-extension pointer (mentioned, not built).
- Commit: `ch05: sections 7-8 the pattern repeats and the handoff`.

### Task 7: Bibliographic Notes
File `bib-notes.tex`. ~0.5 pp.
- Checklist: Elman (1990, finding structure in time); Werbos (1990, BPTT); Hochreiter (1991) / Bengio et al. (1994, the difficulty of learning long-term dependencies, vanishing gradients); Hochreiter and Schmidhuber (1997, LSTM, for the natural-extension pointer). Add new entries to `references.bib`; confirm `\cite`s resolve.
- Commit: `ch05: bibliographic notes and references.bib entries`.

### Task 8: Integration and verification
- Build clean (`make -C book cleanall && make -C book`), zero undefined refs, zero overfull heads.
- Figure placed (caption in PDF). Real-code diff (`RNNCell`) vs source. Numbers reconciled with the notebook.
- **Decision-A audit:** `sec:rnn-alice` contains the full Bengio comparison (the numbers Chapter 4 deferred); `\Cref{ch:fixed-context-lm}` resolves.
- **De-serialization audit:** Chapter 3's experiment is named "the permuted-pixel control" (no "shapes experiment"); no "post" idiom.
- Two axes only; ~8 pp.
- Commit: `ch05: integration pass; build clean, figure placed, comparison absorbed`.

## 9. Risks

| Risk | Mitigation |
|---|---|
| The RNN training (pure Python, BPTT, 15 epochs) is slow in the notebook. | The model is small (`RNNCell(75,64)`); if runtime is a problem, reduce epochs and update numbers, keeping the corpus identical to Chapter 4 for a fair comparison. |
| A fresh seed shifts the 2.06 loss, breaking the Bengio comparison (2.06 vs 2.04). | Notebook-first; if the RNN value shifts, update both the number and the comparison prose (the claim is "comparable endpoint, more epochs and BPTT," not an exact tie). |
| The vanishing-gradient demonstration is hard to show cleanly in pure Python. | Show the Jacobian-norm decay over unroll depth numerically in the notebook; the prose carries the spectral-radius argument. |
| The absorbed comparison duplicates Chapter 4 material. | Chapter 4 only previews; the numbers and the full give-up/gain analysis live here. The part-level review checks the seam. |

## 10. Out of Scope
- LSTM and GRU implementations (vanilla `RNNCell` only; one-line pointer).
- Attention and the Transformer (Chapter 6; forward reference only).
- Exercises.

## 11. Success Criteria
1. A reader understands time-translation equivariance as weight sharing across time, can read the stateful `RNNCell` and its BPTT, and understands vanishing gradients as the price of the prior.
2. The loss trajectory and parameter count regenerate from the notebook; `fig:ch05-rnn-loss` is placed.
3. The full Bengio-versus-RNN comparison lives here (decision A); `\Cref{ch:fixed-context-lm}` resolves; Chapter 4 has no leftover comparison.
4. Chapter 3's experiment is referred to as the permuted-pixel control; backward `\Cref`s resolve; forward ref to Chapter 6 prose-only; zero undefined refs.
5. Two axes only; pure-Python `scratchnn`.

## 12. Next Step
`/bookwright:draft chapter5`. After Chapter 5, Part II concludes with `/bookwright:plan chapter6` (the Transformer capstone: NumPy + PyTorch, the third axis, the heaviest notebook).
