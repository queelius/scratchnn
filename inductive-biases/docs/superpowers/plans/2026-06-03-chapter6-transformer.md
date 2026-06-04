# Implementation Plan: Chapter 6, Attention as Content-Addressable Memory

**Author:** Alexander Towell
**Date:** 2026-06-03
**Status:** Plan (ready for `/bookwright:draft chapter6`)
**Specs:** `docs/superpowers/specs/2026-06-02-master-design.md`, `docs/superpowers/specs/2026-06-03-part2-design.md`
**Base SHA:** branch `inductive-biases-part2` (Chapters 3 to 5 drafted)
**Scope:** Chapter 6 (Part II capstone), ~13 pp plus one paired notebook. The heaviest chapter in the book.

## 1. Goal

Draft Chapter 6, "Attention as Content-Addressable Memory," the architecture-axis capstone. Attention is a learned content-addressable lookup (query matches keys to retrieve values, a pointer dereference); depth is the number of composed lookups. A synthetic pointer-dereferencing task that *requires* content-addressable memory is used to probe the prior, and the scaling investigation delivers the book's third axis: the right architecture is necessary but not sufficient, and whether gradient descent realizes the bias-permitted solution depends on positional-encoding scale, initialization, head count, and depth. This is **implementation realization**, revealed here and carried into Part III.

This is where pure Python ends. The transformer is hand-derived NumPy (`examples/transformer.py`); the chapter says so plainly.

## 2. Architecture Summary

Capstone of Part II; `book/parts/part2.tex` gains `\input{chapters/ch06/ch06}`. Per-file structure under `book/chapters/ch06/`. Sections auto-numbered; `\Cref` resolvable / prose-only forward; `listings` for code (NumPy now, not `scratchnn`). No new preamble needed.

**Code boundary (Part II spec decision C):** the transformer model is NumPy (`examples/transformer.py`: `CausalMultiHeadAttention`, `FFN`, `TransformerBlock`, `Transformer`, `Adam`, sinusoidal positions, softmax, gelu). PyTorch (`examples/pytorch/`) appears only as the M=32 depth sweep, cited not run (see notebook scope). The chapter states the NumPy switch when it arrives and walls off the PyTorch numbers as non-comparable.

## 3. Notebook Scope (the key decision for this chapter)

The Chapter 6 notebook is **NumPy-only; it does not add or use `torch`** (the project uv env has numpy/matplotlib/scikit-learn but not torch, and the book's pure-Python ethos argues against adding it for one figure). It splits experiments into re-run vs cited:

- **Re-run in the notebook (NumPy, tractable):**
  - Experiment 1, single-lookup at M=8: the grokking demonstration (MLP 1.000; 1L transformer stuck ~0.654; 2L groks to 1.000). Generate `fig:ch06-grokking` (the 2L loss/accuracy curve showing the plateau-then-transition).
  - The M-scaling sweep (MLP vs 2L transformer, M in {8,16,24,32,...}): the plateau beyond M=8. Generate `fig:ch06-scaling` (accuracy vs M, both models).
  - The numerical audit (float64 epsilon=1e-6 worst relative error ~2.3e-7, confirming the math is correct and the float32 noise is cancellation).
- **Cited from `examples/RESULTS.md` (not re-run; some need PyTorch or are long):**
  - The four-hypothesis investigation results, the positional-encoding-scale fix (sinusoidal 0.747 to learned-PE 1.000 at M=16), the kitchen-sink recipe, and especially the **PyTorch M=32 depth transition (2L 0.616 vs 3L 0.946)**. Presented as a table (`tab:depth-transition`) with the in-text caveat that PyTorch internals differ from the NumPy model, so its absolute numbers are read only against each other.

If re-running the M-scaling sweep proves too slow, the notebook plots `fig:ch06-scaling` from the recorded data points (cite `RESULTS.md`) and says so. The grokking figure (`fig:ch06-grokking`) is the one figure the chapter most needs as a live receipt and should be regenerated.

This honors the Part II decision B (figures where they add most) within the no-torch constraint: two notebook-generated NumPy figures (grokking, scaling), the PyTorch depth result as a cited table.

## 4. File Structure

```
book/
  parts/part2.tex                          (Task 1: + \input{chapters/ch06/ch06})
  chapters/ch06/
    ch06.tex                               (Task 1: \chapter + opening + \inputs)
    01-lookup-and-task.tex                 (Task 3: sec 1-3)
    02-depth-and-multihead.tex             (Task 4: sec 4-6 + grokking figure)
    03-scaling-and-investigation.tex       (Task 5: sec 7-8 + scaling figure)
    04-fix-and-verification.tex            (Task 6: sec 9-10 + depth table)
    05-three-axes-and-appendix.tex         (Task 7: sec 11 + appendix)
    bib-notes.tex                          (Task 8)
  figures/ch06-grokking.pdf, ch06-scaling.pdf   (Task 2)
notebooks/ch06-transformer.ipynb           (Task 2)
```

## 5. Cross-reference Map

**DEFINED:** `ch:transformer`; `sec:attention-lookup`, `sec:pointer-task`, `sec:one-layer-limit`, `sec:exp-single-lookup`, `sec:multihead`, `sec:exp-multihop`, `sec:scaling-puzzle`, `sec:four-hypotheses`, `sec:pe-scale-fix`, `sec:partial-confirmation`, `sec:three-axes`, `sec:richer-dgp`; `fig:ch06-grokking`, `fig:ch06-scaling`; `tab:depth-transition`.

**Resolvable backward `\Cref` (Chapters 1 to 5 drafted):** `\Cref{ch:foundations}`, `\Cref{sec:inductive-biases}` and `\Cref{sec:parallel-axes}` (the two axes; `sec:three-axes` extends them), `\Cref{ch:cnn}`, `\Cref{ch:rnn}`, `\Cref{ch:fixed-context-lm}` (the prior architectures recap, and the MDL / Solomonoff north star the RL forward-pointer reuses).

**Forward references (PROSE-ONLY):** Chapter 7 (reverse-engineering this trained transformer: induction heads, in-context learning) and Chapter 8 (reinforcement learning; the heads-as-bias frame). Not drafted yet.

**Expected-unresolved baseline: EMPTY.**

## 6. Lessons Inherited

All prior lessons, with extra weight on two given this chapter's number density: **notebook-first with stored outputs** and **regime-reconciliation** (Chapter 6 has the most specific numbers in the book; reconcile every interpreting sentence with the notebook for the re-run experiments, and quote the cited PyTorch numbers verbatim from `RESULTS.md`). Plus: place both figures (S1); real NumPy code from `examples/transformer.py`; clean-rebuild; no em-dashes; per-file header blocks; `\Cref` resolvable / prose-only forward; short chapter title; de-serialize from the booklet `.tex`.

**Chapter-6-specific de-serialization:** reword the pointer to `docs/series/archive/transformer-text.md` (an archived source, not a book chapter); convert any "post" idiom; this chapter reveals the **third axis** (allowed and intended here, unlike Chapters 3 to 5).

## 7. Source Material

| Source | Use |
|---|---|
| `../../docs/booklet/chapters/06-transformer-pointers.tex` | primary port source |
| `../../docs/series/06-transformer-pointers.md` | voice exemplar (blog idiom; do not copy structure) |
| `../../examples/transformer.py` | real NumPy code: attention, MHA, FFN, block, Transformer, Adam |
| `../../examples/simple_pointer_dgp.py` | the pointer DGP |
| `../../examples/RESULTS.md` | the recorded experiment log to CITE (investigations, PE-fix, kitchen-sink, PyTorch M=32 depth) |
| `../../examples/pointer_experiments.py`, `pointer_scaling.py` | the NumPy experiments the notebook re-runs |
| merged Part I + Chapters 3 to 5 | backward `\Cref` targets |

## 8. Tasks

Nine tasks (this chapter is larger). Notebook is Task 2. No exercises.

### Task 1: Scaffold
- `book/parts/part2.tex`: add `\input{chapters/ch06/ch06}` after Chapter 5.
- `book/chapters/ch06/ch06.tex`: header block; `\chapter[Attention as Content-Addressable Memory]{Attention as Content-Addressable Memory}\label{ch:transformer}`; opening prose (the architecture-axis capstone; attention as a learned lookup; pure Python ends here). `\input` the five section files and `bib-notes`.
- Placeholders; compile check.
- Commit: `ch06: scaffold chapter and part wiring`.

### Task 2: Paired notebook (NumPy-only; run first; per Section 3 scope)
`notebooks/ch06-transformer.ipynb`. Port from `examples/pointer_experiments.py`, `pointer_scaling.py`, `transformer.py`, `simple_pointer_dgp.py`.
- Re-run: Experiment 1 grokking at M=8 (MLP 1.000, 1L ~0.654, 2L groks to 1.000); the M-scaling sweep; the float64 numerical audit (~2.3e-7). Fix seeds; stored outputs.
- Generate `book/figures/ch06-grokking.pdf` (the 2L grokking curve) and `book/figures/ch06-scaling.pdf` (accuracy vs M).
- A Results cell records the re-run numbers and explicitly lists the CITED numbers (from `RESULTS.md`) the prose uses: PE-fix (0.747 to 1.000 at M=16), kitchen-sink (M=24 0.998), PyTorch M=32 depth (2L 0.616, 3L 0.946). Do NOT import torch.
- Commit: `ch06: paired notebook (NumPy), seeded and executed; grokking and scaling figures`.

### Task 3: Sections 1 to 3, attention as a lookup, the task, why one layer is not enough
File `01-lookup-and-task.tex`. ~3 pp. Header: DEFINED `sec:attention-lookup`, `sec:pointer-task`, `sec:one-layer-limit`; RESOLVED `\Cref{ch:foundations}`.
- Checklist: query-key-value as content-addressable lookup (pointer dereference); the real NumPy scaled-dot-product-attention listing from `transformer.py`. The 12-bit pointer DGP from `simple_pointer_dgp.py`. Why one attention layer cannot compose two lookups; depth = composed lookups. Mark the NumPy switch (pure Python ends here).
- Commit: `ch06: sections 1-3 attention as a lookup and the pointer task`.

### Task 4: Sections 4 to 6, depth experiments and multi-head
File `02-depth-and-multihead.tex`. ~3 pp. Header: DEFINED `sec:exp-single-lookup`, `sec:multihead`, `sec:exp-multihop`, `fig:ch06-grokking`; RESOLVED (none new).
- Checklist (sec 4): Experiment 1 (single lookup): MLP 1.000, 1L stuck ~0.654, 2L groks to 1.000 (plateau ~0.69 to ~iter 600, sharp drop, machine-zero by ~1500). **Place `fig:ch06-grokking`.** Numbers from the notebook. (sec 5): real NumPy multi-head attention listing; 2L/2H matches 2L/1H at M=8. (sec 6): Experiment 2 multi-hop: 2L 0.977, 3L 0.995.
- Commit: `ch06: sections 4-6 depth experiments and multi-head attention`.

### Task 5: Sections 7 to 8, the scaling puzzle and the investigation
File `03-scaling-and-investigation.tex`. ~2.75 pp. Header: DEFINED `sec:scaling-puzzle`, `sec:four-hypotheses`, `fig:ch06-scaling`; RESOLVED (none new).
- Checklist (sec 7): the transformer plateaus near chance for M>=16 while the MLP wins (scaling table M=8..64). **Place `fig:ch06-scaling`.** (sec 8): the four-hypothesis investigation, CITED from `RESULTS.md`: audit (math correct, float64 2.3e-7, float32 noise is cancellation), optimization (no recipe rescues), supervision (dense hurts; [QUERY] token 0.816), architecture variants (4 heads 0.983; learned PE 1.000). State these are from the recorded runs.
- Commit: `ch06: sections 7-8 the scaling puzzle and the investigation`.

### Task 6: Sections 9 to 10, the fix and the verification
File `04-fix-and-verification.tex`. ~2.25 pp. Header: DEFINED `sec:pe-scale-fix`, `sec:partial-confirmation`, `tab:depth-transition`; RESOLVED (none new).
- Checklist (sec 9): the positional-encoding-scale fix: embedding init 0.02 vs PE magnitude ~1 means the content signal is ~35x too small; learned PE or rescaling. (sec 10): verification, the bias is partially confirmed: learned PE fixes M=16 (0.747 to 1.000) but not M>=24; the kitchen-sink (M=24 0.998); and the **PyTorch M=32 depth transition** (`tab:depth-transition`: 2L 0.616 never transitions vs 3L 0.946, clean transition ~iter 7000), CITED from `RESULTS.md` with the non-comparability caveat (different init/internals from the NumPy model). The lookup-fan-out vs address-decode caveat (M=24 solvable at 2L, M=32 not).
- Commit: `ch06: sections 9-10 the positional-encoding fix and verification`.

### Task 7: Section 11 and the appendix, the three axes
File `05-three-axes-and-appendix.tex`. ~1.5 pp. Header: DEFINED `sec:three-axes`, `sec:richer-dgp`; RESOLVED `\Cref{sec:inductive-biases}`, `\Cref{sec:parallel-axes}`, `\Cref{ch:cnn}`, `\Cref{ch:rnn}`, `\Cref{ch:fixed-context-lm}`.
- Checklist (sec 11): the synthesis. The previous chapters each committed to a structural prior (`\Cref{ch:cnn}` spatial weight sharing, `\Cref{ch:rnn}` recurrent state, `\Cref{ch:fixed-context-lm}` bounded window). Attention is content-addressable. But the right architecture is necessary, not sufficient: realization (PE scale, init, heads, depth) decides whether SGD finds it. **The third axis: implementation realization**, extending `\Cref{sec:inductive-biases}` and `\Cref{sec:parallel-axes}`. Forward (prose) to Chapter 7 (reverse-engineering this model) and Chapter 8 (RL; the heads-as-bias frame; the MDL/Solomonoff north star from `\Cref{ch:fixed-context-lm}`).
- Checklist (appendix, sec:richer-dgp): the Elias-gamma bit-stream DGP that did not train, kept honestly.
- Commit: `ch06: section 11 the three axes, and the appendix`.

### Task 8: Bibliographic Notes
File `bib-notes.tex`. ~0.75 pp.
- Checklist: Vaswani et al. 2017 (`vaswani2017attention`, in refs); the embedding-times-sqrt(d) detail; grokking (Power et al. 2022); the interpretability lineage for the forward pointer (Elhage et al. 2021 `elhage2021framework`, Olsson et al. 2022 `olsson2022induction`, both in refs); the MDL/AIXI thread (`hutter2005uai`). Add `power2022grokking`; confirm `\cite`s resolve.
- Commit: `ch06: bibliographic notes and references.bib entries`.

### Task 9: Integration and verification
- Build clean (`make -C book cleanall && make -C book`), zero undefined refs, zero overfull heads.
- Both notebook figures placed (captions in PDF). Real-code diff (NumPy attention/MHA) vs `examples/transformer.py`. Re-run numbers reconciled with the notebook; cited numbers matched verbatim against `RESULTS.md`.
- **De-serialization audit:** the `transformer-text.md` archive pointer reworded; no "post" idiom. The third axis is revealed here (intended).
- **Code-boundary audit:** the NumPy switch is stated; the PyTorch M=32 result carries its non-comparability caveat.
- ~13 pp.
- Commit: `ch06: integration pass; build clean, figures placed, code boundary marked`.

## 9. Risks

| Risk | Mitigation |
|---|---|
| The NumPy experiments are too slow to re-run in the notebook. | Re-run only Experiment 1 grokking (the essential live receipt) and, if tractable, the M-scaling sweep; cite the rest from `RESULTS.md` and plot `fig:ch06-scaling` from recorded points if needed. |
| The many specific numbers strand the prose if a re-run shifts them. | Notebook-first for the re-run experiments; quote cited numbers verbatim from `RESULTS.md`; reconcile every interpreting sentence. |
| Adding torch to the env to reproduce the M=32 depth result. | Do NOT. Cite it as a table from `RESULTS.md` with the non-comparability caveat (decision in Section 3). |
| The third axis feels abrupt after five chapters of two axes. | `sec:three-axes` builds it from the investigation just completed; it is the payoff, not a bolt-on. |
| Chapter 6 overruns ~13 pp given its density. | Push worked numerical detail into the notebook and the cited tables; keep the prose to the argument. |

## 10. Out of Scope
- Training a real language model or a large transformer (the toy pointer DGP only).
- Reproducing the PyTorch sweeps live (cited from `RESULTS.md`).
- Interpretability of the trained model (Chapter 7) and RL (Chapter 8); forward references only.
- Exercises.

## 11. Success Criteria
1. A reader understands attention as a learned content-addressable lookup, depth as composed lookups, and the pointer task as a probe of that prior.
2. The re-run experiments regenerate from the NumPy notebook; both figures are placed; the cited PyTorch result is presented honestly as non-comparable.
3. The third axis (implementation realization) emerges cleanly from the scaling investigation and sets up Part III.
4. The NumPy switch is explicit; real NumPy listings match `examples/transformer.py`; backward `\Cref`s resolve; forward refs to Chapters 7 and 8 prose-only; zero undefined refs.
5. The archive pointer is reworded; no "post" idiom; ~13 pp.

## 12. Next Step
`/bookwright:draft chapter6`. After Chapter 6, Part II is fully drafted: run `/bookwright:review part2` (the part-level multi-agent review), then `/bookwright:integrate part2`, then merge Part II to `main`.
