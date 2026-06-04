# Implementation Plan: Chapter 4, Fixed-Context Language Models

**Author:** Alexander Towell
**Date:** 2026-06-03
**Status:** Plan (ready for `/bookwright:draft chapter4`)
**Specs:** `docs/superpowers/specs/2026-06-02-master-design.md`, `docs/superpowers/specs/2026-06-03-part2-design.md`
**Base SHA:** branch `inductive-biases-part2` (Chapter 3 drafted)
**Scope:** Chapter 4 (second chapter of Part II), ~7 pp plus one paired notebook.

## 1. Goal

Draft Chapter 4, "Fixed-Context Language Models," by porting and elevating the booklet chapter. Bengio's 2003 model is the simplest neural sequence model: embed a fixed window of N tokens, concatenate, feed a position-sensitive MLP head. The chapter states its hybrid prior (a hard memory cutoff at N, shared embeddings, a position-sensitive head), shows the real pure-Python `Embedding` and `EmbedConcat`, trains a char-level model on Alice, and locates the architecture on the inductive-bias map. It is pure-Python `scratchnn` throughout.

**Defining decision (Part II spec decision A):** the booklet drafted this chapter as if the RNN came first ("the RNN in the previous chapter landed at 2.06"; sections "What Bengio gives up / gains vs the RNN"). In book order the RNN is the next chapter. This plan re-aims those references forward: Chapter 4 keeps a brief forward preview of the recurrent alternative, and the full Bengio-versus-RNN comparison moves to Chapter 5.

## 2. Architecture Summary

Second chapter of Part II; `book/parts/part2.tex` already exists (created for Chapter 3) and gains an `\input{chapters/ch04/ch04}`. Per-file structure under `book/chapters/ch04/`. Sections auto-numbered; `\Cref` for resolvable, prose-only for forward; `listings` for code. Pure-Python `scratchnn`: `Embedding`, `EmbedConcat`, `Linear`, `Tanh`, `Network`, `SoftmaxCrossEntropy`.

## 3. Tech Stack

LaTeX (book, pdflatex + biber, listings). Notebook: Python + uv + Jupyter, `scratchnn` editable, `matplotlib` for the loss-curve figure. Pure-Python `scratchnn` (no NumPy).

## 4. File Structure

```
book/
  parts/part2.tex                          (Task 1: + \input{chapters/ch04/ch04})
  chapters/ch04/
    ch04.tex                               (Task 1: \chapter + opening + \inputs)
    01-arch-and-embedding.tex              (Task 3: sec 1-2)
    02-biases-and-training.tex             (Task 4: sec 3-4)
    03-alice-experiment.tex                (Task 5: sec 5 + figure)
    04-limits-map-handoff.tex              (Task 6: sec 6-8)
    bib-notes.tex                          (Task 7)
  figures/ch04-lm-loss.pdf                 (Task 2: notebook output)
notebooks/ch04-fixed-context-lm.ipynb      (Task 2)
```

## 5. Cross-reference Map

**DEFINED:** `ch:fixed-context-lm`; `sec:bengio-arch`, `sec:embedding-layer`, `sec:bengio-bias`, `sec:bengio-training`, `sec:bengio-alice`, `sec:window-limits`, `sec:bengio-map`, `sec:bengio-handoff`; `fig:ch04-lm-loss`.

**Resolvable backward `\Cref`:** `\Cref{ch:foundations}`, `\Cref{sec:mlp-backprop}` (the MLP head), `\Cref{sec:softmax}` (the categorical output head), `\Cref{ch:cnn}` (the 1-D CNN comparison in the map; Chapter 3 is drafted, so this resolves).

**Forward references (PROSE-ONLY):** Chapter 5 (the recurrent network, "the next chapter"; re-aimed per decision A) and Chapter 6 (the Transformer). Both prose-only; neither is drafted yet.

**Expected-unresolved baseline: EMPTY.**

## 6. Lessons Inherited

All Part I and Chapter 3 lessons: notebook-first with stored outputs; place the figure with `\includegraphics` + `\Cref` (S1); regime-reconciliation if a number shifts; real code from source; clean-rebuild to verify; no em-dashes; per-file header blocks; `\Cref` resolvable / prose-only forward; short chapter title; **two axes only** (no third axis); de-serialize from the booklet `.tex`, never the series `.md`. Chapter-4-specific: the **decision-A re-aim** is a first-class task item (Task 6), and the surviving "The post is next" must go.

## 7. Source Material

| Source | Use |
|---|---|
| `../../docs/booklet/chapters/04-fixed-context-lm.tex` | primary port source |
| `../../docs/series/04-fixed-context-lm.md` | voice exemplar (full of blog idiom; do not copy structure) |
| `../../src/scratchnn/embedding.py` | real code: `Embedding`, `EmbedConcat` |
| `../../examples/fixed_context_lm.py` | notebook experiment (port; if absent, build per the booklet setup) |
| merged Part I + Chapter 3 | backward `\Cref` targets |

## 8. Tasks

Eight tasks. Notebook is Task 2. No exercises.

### Task 1: Scaffold
- `book/parts/part2.tex`: add `\input{chapters/ch04/ch04}` after the Chapter 3 input.
- `book/chapters/ch04/ch04.tex`: header block; `\chapter[Fixed-Context Language Models]{Fixed-Context Language Models}\label{ch:fixed-context-lm}`; opening prose (the simplest neural sequence model: a windowed MLP over embeddings); `\input` the four section files and `bib-notes`.
- Placeholder section files so it compiles.
- **Verify:** `make -C book` compiles; TOC shows Chapter 4; zero undefined refs.
- Commit: `ch04: scaffold chapter and part wiring`.

### Task 2: Paired notebook (run first)
`notebooks/ch04-fixed-context-lm.ipynb`. Port from `examples/fixed_context_lm.py`.
- Char-level Alice (first 30,000 chars, 75-char vocab). Config N=8, d=16, h=64, SGD lr 0.02, batch 1, 4 epochs. Regenerate the loss trajectory (4.32 to ~2.04 nats/char), perplexity (~7.67), parameter count (~14,331), per-epoch samples, and the MDL bits/char (~2.94). `gradient_check` on `EmbedConcat`.
- Fix seeds; execute end to end with stored outputs; Results cell with exact numbers.
- **Figure:** `book/figures/ch04-lm-loss.pdf` (loss per epoch).
- Commit: `ch04: paired notebook, seeded and executed; lock numbers and figure`.

### Task 3: Sections 1 to 2, the architecture and the Embedding layer
File `01-arch-and-embedding.tex`. ~1.75 pp. Header: DEFINED `sec:bengio-arch`, `sec:embedding-layer`; RESOLVED `\Cref{ch:foundations}`, `\Cref{sec:mlp-backprop}`.
- Checklist: embed N tokens, concatenate, feed a position-sensitive MLP head (Bengio 2003). Real `Embedding` (id-to-vector lookup as one-hot `Linear`) and `EmbedConcat` (embed N, concatenate, slice-on-backward) listings from source.
- Commit: `ch04: sections 1-2 the architecture and the Embedding layer`.

### Task 4: Sections 3 to 4, the inductive biases and training
File `02-biases-and-training.tex`. ~1.5 pp. Header: DEFINED `sec:bengio-bias`, `sec:bengio-training`; RESOLVED `\Cref{sec:softmax}`.
- Checklist (sec 3): the hybrid prior, a hard memory cutoff at N, shared embeddings, and a position-sensitive head (concatenation places token i in a fixed slice so the head learns position-specific weights). Two axes only. (sec 4): pure-Python `net.fit` on char-level data; the categorical head via `\Cref{sec:softmax}`.
- Commit: `ch04: sections 3-4 the inductive biases and training`.

### Task 5: Section 5, the experiment (char-level Alice)
File `03-alice-experiment.tex`. ~1.25 pp. Header: DEFINED `sec:bengio-alice`, `fig:ch04-lm-loss`; RESOLVED (none new).
- Checklist: the char-level Alice run; loss 4.32 to 2.04 nats/char, perplexity 7.67, ~14,331 params; per-epoch samples quoted; the MDL note (2.04 nats/char is ~2.94 bits/char), the chapter's Solomonoff/compression north star. **Place `fig:ch04-lm-loss`** with `\includegraphics` and a `\Cref`. Numbers from the notebook.
- Commit: `ch04: section 5 the char-level Alice experiment`.

### Task 6: Sections 6 to 8, bounded-window limits, the map, the handoff (decision-A re-aim)
File `04-limits-map-handoff.tex`. ~1.75 pp. Header: DEFINED `sec:window-limits`, `sec:bengio-map`, `sec:bengio-handoff`; RESOLVED `\Cref{ch:cnn}`.
- Checklist (sec 6, `sec:window-limits`): the fixed window cannot reach past N tokens. **Decision A:** give a brief FORWARD preview of the recurrent alternative (the next chapter, prose-only, not `\Cref`) and the 1-D CNN (`\Cref{ch:cnn}`). Do NOT do the full Bengio-vs-RNN comparison here; that moves to Chapter 5. The booklet's separate "gives up vs the RNN" and "gains over the RNN" sections are merged into this one forward-looking section.
- Checklist (sec 7, `sec:bengio-map`): the inductive-bias map table (MLP / 1-D CNN / Bengio / RNN / Transformer x position-handling x memory-horizon). The RNN and Transformer rows are forward, framed as such.
- Checklist (sec 8, `sec:bengio-handoff`): forward-promise the architecture progression toward attention (prose). **Fix "The post is next."**
- Commit: `ch04: sections 6-8 bounded-window limits, the map, the handoff`.

### Task 7: Bibliographic Notes
File `bib-notes.tex`. ~0.5 pp.
- Checklist: Bengio et al. 2003 (`bengio2003nplm`, already in references.bib); Shannon (the entropy-of-English / bits-per-char lineage) and the MDL / Solomonoff connection (reuse `hutter2005uai`, already present, for the compression-as-prediction north star); optionally Goodfellow. Add any new entries; confirm `\cite`s resolve.
- Commit: `ch04: bibliographic notes and references.bib entries`.

### Task 8: Integration and verification
- Build clean (`make -C book cleanall && make -C book`), zero undefined refs, zero overfull heads.
- Figure placed (caption in PDF). Real-code diff (`Embedding`, `EmbedConcat`) vs source. Numbers reconciled with the notebook.
- **Decision-A audit:** no "the RNN in the previous chapter" survives; the RNN is forward-referenced (prose-only); the full comparison is deferred to Chapter 5 (leave a clear hook). No "post" idiom.
- Two axes only; ~7 pp.
- Commit: `ch04: integration pass; build clean, figure placed, RNN refs re-aimed`.

## 9. Risks

| Risk | Mitigation |
|---|---|
| The decision-A re-aim drops the Bengio-vs-RNN comparison entirely. | Chapter 4 keeps the forward preview and the map row; Chapter 5's plan (`sec:rnn-alice`) owns the full comparison. The part-level review checks the seam. |
| A fresh seed shifts the loss / perplexity. | Notebook-first; write prose to the regenerated values. |
| Pure-Python char-LM training is slow. | 4 epochs on 30k chars as specified; if slow, reduce and update numbers, keeping the comparison fair. |
| `examples/fixed_context_lm.py` differs from the planned config. | Use it as the basis; reconcile to the seeded notebook's numbers. |

## 10. Out of Scope
- Word-level or large-corpus language modeling (char-level Alice only).
- The full RNN comparison (Chapter 5) and attention (Chapter 6); forward references only.
- Exercises.

## 11. Success Criteria
1. A reader understands Bengio's windowed-MLP-over-embeddings model and its hybrid prior, and can read the real `Embedding` / `EmbedConcat` code.
2. The loss/perplexity numbers regenerate from the notebook; `fig:ch04-lm-loss` is placed.
3. Decision A holds: no "RNN in the previous chapter"; the RNN is a forward, prose-only preview; the comparison is deferred to Chapter 5.
4. Backward `\Cref`s (`ch:foundations`, `sec:softmax`, `ch:cnn`) resolve; forward refs prose-only; zero undefined refs.
5. Two axes only; pure-Python `scratchnn`.

## 12. Next Step
`/bookwright:draft chapter4`. After Chapter 4, Part II continues with `/bookwright:plan chapter5` (RNN), which absorbs the full Bengio-vs-RNN comparison.
