# Implementation Plan: Chapter 7, Reverse-Engineering the Pointer Transformer

**Author:** Alexander Towell
**Date:** 2026-06-05
**Status:** Plan (ready for `/bookwright:draft chapter7`)
**Specs:** `docs/superpowers/specs/2026-06-02-master-design.md`, `docs/superpowers/specs/2026-06-04-part3-design.md`
**Base SHA:** `30d5cae` (branch `inductive-biases-part3`)
**Scope:** Chapter 7 (opens Part III), ~13 pp plus one paired notebook. The book's flagship chapter.

## 1. Goal

Draft Chapter 7, "Reverse-Engineering the Pointer Transformer." It opens Part III and delivers the book's flagship honest-empiricism result. The chapter takes Chapter 6's trained pointer transformer apart and reads the architectural bias back out of the weights. Two acts: the M=8 model (NumPy) confirms Chapter 6's prediction cleanly (layer 1 aggregates the address, layer 2 dereferences, ablations break it), and the M=32 model (PyTorch) refutes the obvious story (the attention maps look illegible, two pre-registered hypotheses fail) until a causal probe proves it is a perfect pointer. The meta-lesson: attention weight is not information flow. This is the empirical payoff of the third axis (implementation realization).

## 2. Architecture Summary

Chapter 7 opens Part III, so its scaffold creates `book/parts/part3.tex` (`\part{Interpretation and Beyond}` + the booklet blurb) and adds `\input{parts/part3.tex}` to `book.tex` after Part II. Per-file structure under `book/chapters/ch07/`. Sections auto-numbered; `\Cref` resolvable / prose-only forward; `listings` for code. Two models, two frameworks (decision D of the Part III spec): NumPy for M=8 (`examples/pointer_interp.py`), PyTorch for M=32 (`examples/pytorch/pointer_interp_deep.py`, loading the committed checkpoint `interp_deep_M32.pt`).

## 3. Tech Stack

LaTeX (book, pdflatex + biber, listings). Notebook: Python + uv + Jupyter; NumPy for the M=8 probes; **`torch` for the M=32 probes** (added to `pyproject.toml`; loads the checkpoint, no training); `matplotlib` for the three figures.

## 4. File Structure

```
book/
  book.tex                                 (Task 1: + \input{parts/part3.tex})
  parts/part3.tex                          (Task 1: \part divider + blurb + \input ch07)
  chapters/ch07/
    ch07.tex                               (Task 1: \chapter + opening + \inputs)
    01-setup-and-circuit.tex               (Task 3: sec 1-2)
    02-m8-analysis.tex                      (Task 4: sec 3-5, the M=8 NumPy act)
    03-induction-and-icl.tex               (Task 5: sec 6-8)
    04-scaling-the-lens.tex                (Task 6: sec 9, the flagship; 3 figures)
    05-closing.tex                         (Task 7: sec 10)
    bib-notes.tex                          (Task 8)
  figures/ch07-attn-contrast.pdf, ch07-lookup-by-address.pdf, ch07-causal-trace.pdf  (Task 2)
notebooks/ch07-interpretability.ipynb      (Task 2)
```

## 5. Cross-reference Map

**DEFINED:** `ch:interpretability`; `sec:interp-setup`, `sec:circuit-restated`, `sec:layer2-deref`, `sec:layer1-aggregator`, `sec:ablations`, `sec:induction-head`, `sec:icl`, `sec:small-model-limits`, `sec:scaling-the-lens`, `sec:interp-closing`; `fig:attn-contrast`, `fig:lookup-by-address`, `fig:causal-trace` (all three in `sec:scaling-the-lens`).

**Resolvable backward `\Cref` (Parts I-II merged):** `\Cref{ch:transformer}`, `\Cref{sec:one-layer-limit}` (the two-stage necessity argument it tests), `\Cref{sec:pe-scale-fix}` (the M=32 init aside ties to the PE-scale lesson), `\Cref{sec:parallel-axes}` (multi-head), `\Cref{sec:three-axes}` (the closing completes the frame).

**Forward references (PROSE-ONLY):** Chapter 8 (RL, "the closing chapter"). Not drafted yet.

**Expected-unresolved baseline: EMPTY.**

## 6. Lessons Inherited

All prior lessons. Chapter-7-specific:
- **torch for the M=32 probes** (decision C): add to `pyproject.toml`; load `interp_deep_M32.pt`; run the probes (cheap, no training). The M=8 cells are NumPy.
- **Regenerate the curated figure set** (decision A): `fig:causal-trace` (flagship), `fig:attn-contrast` (M=8 legible vs M=32 illegible), `fig:lookup-by-address`. From the notebook, placed with `\Cref`.
- **Two-models clarity** (decision D): §1-8 is the NumPy M=8 model; §9 is a fresh introspectable PyTorch M=32 model, not the Chapter 6 sweep model; accuracies not comparable. State it.
- **Heavy de-serialization** (decision E): no "post-6"/"Post 6"; fix the garbled §10; reword the archived `transformer-text.md` pointer.
- **Regime-reconciliation:** the M=8 model retrains each run; fix the seed and reconcile the probe numbers (the qualitative claims are robust, the exact weights may move). The M=32 numbers come from the fixed checkpoint, so they are stable; quote them.

## 7. Source Material

| Source | Use |
|---|---|
| `../../docs/booklet/chapters/07-interpretability.tex` | primary port source |
| `../../docs/series/07-interpretability.md` | voice exemplar (blog idiom; do not copy) |
| `../../examples/pointer_interp.py` | M=8 NumPy probes (attention inspection, ablations) |
| `../../examples/pytorch/pointer_interp_deep.py` + `interp_deep_M32.pt` | M=32 PyTorch probes (flip test, ablations, causal trace) |
| `../../examples/RESULTS.md` | the recorded interp numbers to reconcile against |
| `../../docs/series/figures/07-interp{,-deep}/` | reference PNGs (regenerate, do not port) |
| merged Parts I-II | backward `\Cref` targets |

## 8. Tasks

Nine tasks. Notebook is Task 2. No exercises.

### Task 1: Scaffold (open Part III, chapter wrapper)
- `book/parts/part3.tex`: `\part{Interpretation and Beyond}` + the ported booklet blurb, then `\input{chapters/ch07/ch07}`.
- `book/book.tex`: add `\input{parts/part3.tex}` after Part II.
- `book/chapters/ch07/ch07.tex`: header block; `\chapter[Reverse-Engineering the Pointer Transformer]{Reverse-Engineering the Pointer Transformer}\label{ch:interpretability}`; opening prose (the lens turns on the trained model; the third axis made empirical); `\input` the five section files and `bib-notes`.
- Placeholders; compile check.
- Commit: `ch07: scaffold Part III, chapter wrapper, part wiring`.

### Task 2: Paired notebook (run first; NumPy M=8 + PyTorch M=32)
`notebooks/ch07-interpretability.ipynb`. Add `torch` to `pyproject.toml` first (`uv sync`).
- M=8 (NumPy, `pointer_interp.py`): retrain 2L/1H d=32 (seed-fixed, 2000 iters); regenerate layer-2 dereference weights (0.66 to 0.81 on m_a), layer-1 address attention (93 to 99 percent), the ablation table (baseline 1.000; L1-uniform 0.658; L2-uniform 0.636; shuffle-address 0.776).
- M=32 (PyTorch, `pointer_interp_deep.py`): load `interp_deep_M32.pt` (no training); causal flip test (0.995 / 0.005); per-layer ablations (0.540 / 0.763 / 0.600); per-head dereference weight by address (L3H0 cleanly covers 10 of 32); the activation-patching causal trace (embedding 1.000 to L3-readout 1.000).
- Figures: `book/figures/ch07-attn-contrast.pdf` (M=8 legible vs M=32 illegible), `ch07-lookup-by-address.pdf`, `ch07-causal-trace.pdf`.
- Fix seeds; execute end to end (M=8 + M=32); stored outputs; Results cell.
- Commit: `ch07: paired notebook (NumPy M=8 + PyTorch M=32), seeded; three figures`.

### Task 3: Sections 1 to 2, the setup and the circuit restated
File `01-setup-and-circuit.tex`. ~1.5 pp. Header: DEFINED `sec:interp-setup`, `sec:circuit-restated`; RESOLVED `\Cref{ch:transformer}`, `\Cref{sec:one-layer-limit}`.
- Checklist: the plan (read Chapter 6's prediction off the weights); restate the two-stage necessity argument (aggregate the address, then dereference) from `\Cref{sec:one-layer-limit}`. State the two-models framing (decision D): M=8 NumPy here, M=32 PyTorch in §9.
- Commit: `ch07: sections 1-2 the setup and the circuit restated`.

### Task 4: Sections 3 to 5, the M=8 analysis (the confirmatory act)
File `02-m8-analysis.tex`. ~3.5 pp. Header: DEFINED `sec:layer2-deref`, `sec:layer1-aggregator`, `sec:ablations`; RESOLVED (none new); references `fig:attn-contrast` (defined in 04).
- Checklist (sec 3): layer 2's argmax is the address; averaged weight on m_a is 0.66 to 0.81. (sec 4): layer 1 puts 93 to 99 percent on the address positions, MSB first. (sec 5): the ablation table (1.000; 0.658; 0.636; 0.776) causally confirms the prediction. Real NumPy probe listings from `pointer_interp.py`. Numbers from the notebook. Forward-reference the M=8 panel of `\Cref{fig:attn-contrast}`.
- Commit: `ch07: sections 3-5 the M=8 circuit analysis`.

### Task 5: Sections 6 to 8, induction heads, ICL, limits
File `03-induction-and-icl.tex`. ~2.5 pp. Header: DEFINED `sec:induction-head`, `sec:icl`, `sec:small-model-limits`; RESOLVED `\Cref{sec:parallel-axes}`.
- Checklist: the QK/OV reading and the relation to induction heads (Olsson) and the circuits framework (Elhage); ICL as composed lookups; honest limits of the small model. Reword the archived `transformer-text.md` pointer.
- Commit: `ch07: sections 6-8 induction heads, in-context learning, limits`.

### Task 6: Section 9, scaling the lens (THE FLAGSHIP)
File `04-scaling-the-lens.tex`. ~4.5 pp. Header: DEFINED `sec:scaling-the-lens`, `fig:attn-contrast`, `fig:lookup-by-address`, `fig:causal-trace`; RESOLVED `\Cref{sec:pe-scale-fix}`.
- Checklist: the honest arc. The M=32 PyTorch model (fresh introspectable, decision D). Hypothesis 1 (layer 3 refines a layer-2 lookup): REFUTED (layer 2 does almost no dereferencing, 0.047). Hypothesis 2 (heads partition the address space): REFUTED (clean coverage only 10 of 32). The causal flip test (0.995 / 0.005) and the activation-patching causal trace (m_a transported to the readout: embedding 1.000 to L3 readout 1.000) prove a perfect pointer. "Attention weight is not information flow." The Kaiming-vs-Xavier init aside ties to `\Cref{sec:pe-scale-fix}` (implementation realization recurs). **Place all three figures.** Real PyTorch probe listings from `pointer_interp_deep.py` (causal_flip_test, causal_trace). Numbers from the checkpoint (stable).
- Commit: `ch07: section 9 scaling the lens, the causal-probe result`.

### Task 7: Section 10, closing
File `05-closing.tex`. ~1 pp. Header: DEFINED `sec:interp-closing`; RESOLVED `\Cref{sec:three-axes}`; FORWARD (prose) Chapter 8.
- Checklist: the interpretability axis sits across all three axes (`\Cref{sec:three-axes}`); the meta-lesson (causal probes beat weight-reading). Forward-promise Chapter 8 (RL) in prose. Fix the garbled §10 ("The the book identified three axes") and the "closing post" idiom.
- Commit: `ch07: section 10 closing, the inductive-bias frame completed`.

### Task 8: Bibliographic Notes
File `bib-notes.tex`. ~0.5 pp.
- Checklist: Olsson et al. 2022 (induction heads, `olsson2022induction`), Elhage et al. 2021 (circuits framework, `elhage2021framework`), Wang et al. 2022 (IOI circuit, `wang2022wild`) all already in `references.bib`; the activation-patching / causal-tracing lineage (add `meng2022rome` if used). Confirm `\cite`s resolve.
- Commit: `ch07: bibliographic notes`.

### Task 9: Integration and verification
- Build clean (`make -C book cleanall && make -C book`), zero undefined refs, zero overfull heads.
- All three figures placed (captions in PDF). Real-code diff (NumPy + PyTorch probe listings) vs source. Numbers reconciled with the notebook (M=8) and checkpoint (M=32).
- **Two-models audit:** the prose states the M=8 NumPy / M=32 PyTorch distinction; accuracies not cross-compared.
- **De-serialization audit:** no "post"/"Post 6"; §10 ungarbled; archive pointer reworded.
- ~13 pp.
- Commit: `ch07: integration pass; build clean, three figures placed`.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Adding torch breaks the env or the build. | torch is notebook-only (the book build does not need it); pin a CPU wheel; the checkpoint loads in seconds. |
| The M=8 retrain shifts the probe numbers. | Notebook-first; fix the seed; reconcile prose to the run; the qualitative claims (layer 1 aggregates, layer 2 dereferences, ablations break it) are robust. |
| The checkpoint `interp_deep_M32.pt` does not load (version skew). | Verify load in Task 2 before drafting §9; if it fails, pin the torch version that wrote it, or retrain once with `--retrain` (expensive, last resort). |
| §9 is the largest section in the book and could sprawl. | Keep the three figures carrying the argument; the refuted-then-corrected arc is the spine; push raw tables to the notebook. |
| The flagship "wrong story" reads as hindsight-tidy. | Present the two hypotheses as pre-registered and genuinely refuted (the booklet already does this); keep the messy intermediate readings. |

## 10. Out of Scope (Chapter 7)
- A general interpretability-methods survey (these specific probes on this specific model).
- Retraining the M=32 model (load the checkpoint; retrain only if it will not load).
- Chapter 8 / RL content (forward reference only).
- Exercises.

## 11. Success Criteria
1. A reader sees the M=8 prediction confirmed (layer roles + ablations) and the M=32 obvious story refuted then corrected by a causal probe, and takes the meta-lesson.
2. All three figures are placed; the M=8 numbers regenerate from the notebook, the M=32 numbers from the fixed checkpoint.
3. The two-models framing is explicit; backward `\Cref`s resolve; the forward ref to Chapter 8 is prose-only; zero undefined refs.
4. The blog idiom is gone (no "post-6", no garbled §10); the archive pointer is reworded.
5. The third axis is completed empirically; the chapter sets up Chapter 8's "beyond supervised" close.

## 12. Next Step
`/bookwright:draft chapter7` to execute this plan (scaffold + notebook first, then prose, then integration). After Chapter 7, Part III concludes with `/bookwright:plan chapter8` (the book's closing chapter).
