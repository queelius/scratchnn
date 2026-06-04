# Implementation Plan: Chapter 2, Output Heads as Inductive Bias

**Author:** Alexander Towell
**Date:** 2026-06-03
**Status:** Plan (ready for `/bookwright:draft chapter2`)
**Specs:** `docs/superpowers/specs/2026-06-02-master-design.md`, `docs/superpowers/specs/2026-06-02-part1-design.md`
**Base SHA:** `45aed18` (main, with Chapter 1 merged)
**Scope:** Chapter 2 only (second and final chapter of Part I), ~10 pp plus one paired notebook.

## 1. Goal

Draft Chapter 2, "Output Heads as Inductive Bias," by porting and elevating the booklet chapter. The chapter develops the first inductive bias axis (the output head): every supervised network is a maximum-likelihood estimator under an assumed output distribution, the canonical-link theorem explains why the `p - y` gradient recurs, and two new worked heads (Poisson for counts, heteroscedastic Gaussian for input-dependent uncertainty) show the payoff. It closes the supervised-learning frame and hands off to the architecture axis (Part II).

This is the second re-homing in Part I. Chapter 1 is now merged into `main`, so Chapter 2's backward cross-references resolve at build time (a change from Chapter 1, whose cross-chapter refs were all prose-only forward refs).

## 2. Architecture Summary

Same as Chapter 1 (see that plan). Sections auto-numbered (`secnumdepth=2`), `\Cref` for resolvable targets, kebab-case typed labels, `listings` for code. New this chapter: a single theorem environment (the canonical-link theorem), so the scaffold adds `amsthm` and `\newtheorem`. Per-file structure under `book/chapters/ch02/`, `\input` by a `ch02.tex` wrapper, which `book/parts/part1.tex` inputs after Chapter 1.

The Makefile dependency bug found during Chapter 1 is already fixed on `main` (it now depends on `chapters/*/*.tex` via `$(wildcard)`), so incremental builds pick up chapter edits. Integration (Task 10) still verifies on a clean rebuild.

## 3. Tech Stack

LaTeX (book, pdflatex + biber, listings, amsthm). Notebook: Python + uv + Jupyter, `scratchnn` editable, NumPy and matplotlib for the two experiments and their figures. The loss classes themselves are pure-Python `scratchnn`.

## 4. File Structure

```
book/
  parts/part1.tex                          (Task 1: + \input chapters/ch02/ch02)
  preamble.tex                             (Task 1: + amsthm, \newtheorem{theorem})
  chapters/ch02/
    ch02.tex                               (Task 1: \chapter + opening + \inputs)
    01-frame-and-catalogue.tex             (Task 3: sec 1-2)
    02-canonical-link.tex                  (Task 4: sec 3-4, the theorem + three pairings)
    03-poisson.tex                         (Task 5: sec 5)
    04-heteroscedastic.tex                 (Task 6: sec 6)
    05-link-likelihood-mdn.tex             (Task 7: sec 7-8)
    06-synthesis-library-handoff.tex       (Task 8: sec 9-11)
    bib-notes.tex                          (Task 9)
  figures/ch02-poisson.pdf                 (Task 2: notebook output)
  figures/ch02-heteroscedastic.pdf         (Task 2: notebook output)
notebooks/ch02-output-heads.ipynb          (Task 2)
```

No exercises file (book has none). Bibliographic Notes are an unnumbered `\section*`.

## 5. Cross-reference Map

**Labels DEFINED in Chapter 2:**

| Label | File | Kind |
|---|---|---|
| `ch:output-heads` | ch02.tex | chapter |
| `sec:mle-frame` | 01-frame-and-catalogue | section |
| `sec:catalogue` | 01-frame-and-catalogue | section |
| `tab:head-catalogue` | 01-frame-and-catalogue | the five-pairings table |
| `sec:canonical-link` | 02-canonical-link | section |
| `thm:canonical-link` | 02-canonical-link | theorem |
| `eq:expfam-grad` | 02-canonical-link | the NLL gradient = expected minus observed |
| `sec:three-pairings` | 02-canonical-link | section |
| `sec:poisson` | 03-poisson | section |
| `fig:ch02-poisson` | 03-poisson | figure |
| `sec:heteroscedastic` | 04-heteroscedastic | section |
| `fig:ch02-heteroscedastic` | 04-heteroscedastic | figure |
| `sec:link-likelihood` | 05-link-likelihood-mdn | section |
| `sec:mdn` | 05-link-likelihood-mdn | section |
| `sec:parallel-axes` | 06-synthesis-library-handoff | section |
| `sec:library-additions` | 06-synthesis-library-handoff | section |
| `sec:handoff` | 06-synthesis-library-handoff | section |

**RESOLVABLE backward `\Cref` to merged Chapter 1 (these now resolve, unlike Chapter 1's forward refs):**
- `sec:mle-frame` and `sec:three-pairings` to `\Cref{ch:foundations}`
- `sec:canonical-link` to `\Cref{sec:logistic}` and `\Cref{sec:softmax}` (proves their `p - y` "no coincidence")
- `sec:parallel-axes` to `\Cref{sec:inductive-biases}` (completes the synthesis Chapter 1 previewed)
- `sec:mdn` to `\Cref{sec:autograd-note}` (the "natural extension, not built" template)

**FORWARD references (PROSE-ONLY, not `\Cref`):** Chapter 3 (CNN) and the rest of Part II, Chapter 8 (RL). Named by descriptive title. They become `\Cref` only once those chapters are drafted.

**Expected-unresolved baseline for Chapter 2: NONE.** Backward refs to Chapter 1 resolve; forward refs are prose-only. A clean `make` must show zero "Reference undefined."

**Labels Chapter 2 is referenced BY (future):** Part II chapters will `\Cref{ch:output-heads}` and `\Cref{sec:parallel-axes}` when composing a body with a head. Do not rename these.

## 6. Lessons Inherited (from the Chapter 1 plan and its review)

- **Notebook-first** (Part I spec decision C). Chapter 2 is the sharpest case: the prose asserts specific numbers (Poisson minimum-rate 0.068 vs 0.310, the heteroscedastic prediction table, test Gaussian NLL 0.745 vs 0.509, gradient-check residuals near 1e-11). Run the notebook, fix seeds, and write the prose to the regenerated values. If a fresh seed shifts a number, update the prose, do not keep the booklet's old value (this is the Chapter 1 UCI-digits lesson: load_digits gave 94.4/97.2, not the booklet's 95/96, and the prose was corrected to match).
- **Figures must be placed, not just generated** (Chapter 1 finding S1). The notebook saving `figures/ch02-poisson.pdf` is not enough: the Poisson and heteroscedastic section tasks must `\includegraphics` their figure with a `\Cref`, and Task 10 verifies the caption text appears in the built PDF.
- **Real code only.** `PoissonNLLLoss`, `GaussianNLLLoss`, and `softplus` listings come verbatim (trimmed) from `../../src/scratchnn/neural_net.py`. Verified present on `main`.
- **Voice.** No em-dashes (the soul hook blocks them on write, and it fires per edit). Avoid LLM filler; the soul banned-phrase list is authoritative. Match the series voice.
- **Cross-ref discipline.** Per-file header blocks (DEFINED / RESOLVED / FORWARD). `\Cref` for resolvable, prose-only for forward chapters.
- **Build verification.** Trust the now-fixed Makefile for drafting, but Task 10 does a clean rebuild and distinguishes overfull `\hbox` (real) from benign underfull `\vbox`. Give the chapter a short title (`\chapter[Output Heads]`) up front to avoid the running-head overflow that Chapter 1 hit.

## 7. Source Material

| Source | Use |
|---|---|
| `../../docs/booklet/chapters/02-output-heads.tex` | primary port source (structure and prose) |
| `../../docs/series/02-output-heads.md` | prose seed and voice exemplar |
| `../../src/scratchnn/neural_net.py` | real code: `MSELoss`, `PoissonNLLLoss`, `GaussianNLLLoss`, `softplus`, the `Loss` interface, `gradient_check` |
| `../../examples/poisson_regression.py`, `../../examples/heteroscedastic.py` | notebook experiments (locate; if absent, build them in the notebook per the booklet's specified setups) |
| merged Chapter 1 (`book/chapters/ch01/`) | the backward `\Cref` targets; verify each resolves |

## 8. Tasks

Ten tasks. Notebook is Task 2 (notebook-first). No exercises task.

### Task 1: Scaffold (chapter wrapper, part wiring, theorem preamble)

- `book/chapters/ch02/ch02.tex`: chapter header block; `\chapter[Output Heads]{Output Heads as Inductive Bias}\label{ch:output-heads}`; opening prose (port the booklet's one-sentence-thesis opening: every supervised net is an MLE under an assumed output distribution). DE-SERIALIZE the opening: the booklet says "The CNN chapter took the architecture axis on its first non-trivial step", treating CNN as prior work. In book order CNN is the NEXT chapter, so reframe to name only what Chapter 1 established (the two axes) and forward-reference the architecture axis in prose. Then `\input` the six section files and bib-notes.
- `book/parts/part1.tex`: add `\input{chapters/ch02/ch02}` after the Chapter 1 input.
- `book/preamble.tex`: add `\usepackage{amsthm}` and `\newtheorem{theorem}{Theorem}[chapter]` for the one canonical-link theorem.
- Short chapter title set (`[Output Heads]`) to prevent running-head overflow.
- Placeholder section files (header block + `\section{...}\label{...}`) so the chapter compiles.
- **Verify:** `make -C book` compiles; TOC shows Chapter 2 under Part I; zero undefined refs.
- Commit: `ch02: scaffold chapter, part wiring, theorem preamble`.

### Task 2: Paired notebook (run first, lock the numbers and figures)

`notebooks/ch02-output-heads.ipynb`. Port from `examples/poisson_regression.py` and `examples/heteroscedastic.py` plus `scratchnn`.

- Poisson cell: lambda(x) = max(0.1, 2 + 5 sin(pi x)) on [0, 2]; identical 1-16-1 Tanh bodies, MSE head vs Poisson head. Record minimum predicted rate on a grid (target ~0.068 MSE, ~0.310 Poisson) and comparative Poisson NLL. (Erratum 2026-06-03: the executed run produced 0.152 / 0.139, both positive; the chapter uses these regenerated values.) Save `book/figures/ch02-poisson.pdf` (true rate, MSE fit, Poisson fit).
- Heteroscedastic cell: y = sin(x) + noise, sigma(x) = |x|/3 + 0.1 on [0, 6]; 1-16-1 (MSE) vs 1-16-2 (Gaussian NLL). Record the six-row prediction table and test Gaussian NLL (target 0.745 MSE+global-sigma vs 0.509 het). Save `book/figures/ch02-heteroscedastic.pdf` (predictions with a +/- sigma band vs MSE point predictions).
- gradient_check cells for `MSELoss`, `PoissonNLLLoss`, `GaussianNLLLoss` (anchor on a Tanh MLP; residuals near 1e-10/1e-11).
- Fix and record seeds. Execute end to end. A Results markdown cell captures the exact numbers for the prose.
- **Verify:** notebook runs clean; both figures written; Results cell populated.
- Commit: `ch02: paired notebook, seeded and executed; lock numbers and figures`.

### Task 3: Sections 1 to 2, the unifying frame and the catalogue

File `01-frame-and-catalogue.tex`. ~1.75 pp. Header: DEFINED `sec:mle-frame`, `sec:catalogue`, `tab:head-catalogue`; RESOLVED `\Cref{ch:foundations}`.

- Checklist: every supervised loss is an NLL, L = -(1/N) sum log p(y|x); the network parameterizes p(y|x); the head is link plus assumed distribution; tie to Chapter 1's "network emits logits, loss interprets them" via `\Cref{ch:foundations}`. The five-pairings catalogue table (`tab:head-catalogue`): Gaussian/identity/MSE, Bernoulli/logit/BCE, Categorical/softmax/CE, Poisson/log/PoissonNLL, Gamma/reciprocal/GammaNLL, with canonical links. Note the first three are what Chapter 1 built.
- Commit: `ch02: sections 1-2 the MLE frame and the head catalogue`.

### Task 4: Sections 3 to 4, the canonical-link theorem and the three pairings

File `02-canonical-link.tex`. ~2.5 pp. Header: DEFINED `sec:canonical-link`, `thm:canonical-link`, `eq:expfam-grad`, `sec:three-pairings`; RESOLVED `\Cref{sec:logistic}`, `\Cref{sec:softmax}`, `\Cref{ch:foundations}`.

- Checklist (sec 3): exponential family natural form p(y|eta) = h(y) exp(eta T(y) - A(eta)); the identity A'(eta) = E[T(y)]; with the canonical link (network output = natural parameter), the NLL gradient is the expected sufficient statistic minus the observed (label `eq:expfam-grad`). State this as `\begin{theorem}\label{thm:canonical-link}` with a short `proof`. Explicitly: this proves Chapter 1's "no coincidence" `p - y` from `\Cref{sec:logistic}` and `\Cref{sec:softmax}`; MSE is another instance; Poisson (`\Cref{sec:poisson}`) will be a third.
- Checklist (sec 4): brief recap of the three Chapter 1 pairings (identity/Gaussian/MSE, logit/Bernoulli/BCE, softmax/Categorical/CE), each `\Cref`'d back into Chapter 1. One pattern, three objects.
- Commit: `ch02: sections 3-4 the canonical-link theorem and three pairings`.

### Task 5: Section 5, count data (Poisson)

File `03-poisson.tex`. ~1.5 pp. Header: DEFINED `sec:poisson`, `fig:ch02-poisson`; RESOLVED `\Cref{thm:canonical-link}`.

- Checklist: log link, lambda = e^z; NLL = e^z - yz (dropping log y!); gradient e^z - y, the third instance of `\Cref{thm:canonical-link}`; real `PoissonNLLLoss` listing from source. The worked experiment with honest findings (both heads fit a benign target; the structural argument is load-bearing; minimum-rate 0.068 vs 0.310 from the notebook). **Place `fig:ch02-poisson`** with `\includegraphics` and a `\Cref`. Numbers from the notebook.
- Commit: `ch02: section 5 count data, the Poisson head`.

### Task 6: Section 6, heteroscedastic Gaussian

File `04-heteroscedastic.tex`. ~1.75 pp. Header: DEFINED `sec:heteroscedastic`, `fig:ch02-heteroscedastic`; RESOLVED (none new).

- Checklist: the two-output head; mu = z_mu, sigma = softplus(z_s); NLL = (y-mu)^2/(2 sigma^2) + log sigma; hand-derived gradients through softplus; real `GaussianNLLLoss` and `softplus` listings from source. The prediction table and test NLL 0.745 vs 0.509 (the 0.24-nat calibration premium), all from the notebook. **Place `fig:ch02-heteroscedastic`** with `\includegraphics` and a `\Cref` (predictions with uncertainty band). This is the chapter's cleanest "the head buys something tangible" demonstration.
- Commit: `ch02: section 6 heteroscedastic Gaussian and calibrated uncertainty`.

### Task 7: Sections 7 to 8, link/likelihood independence and the MDN pointer

File `05-link-likelihood-mdn.tex`. ~1.5 pp. Header: DEFINED `sec:link-likelihood`, `sec:mdn`; RESOLVED `\Cref{sec:autograd-note}`.

- Checklist (sec 7): same link (sigmoid), two likelihoods (Gaussian-on-output vs Beta); breaks the catalogue's apparent one-to-one. Beta NLL is a **natural extension** (needs only `math.lgamma`), not a library gap. DE-SERIALIZE: the booklet calls this "a reader exercise"; reframe as a natural extension, matching the no-exercises decision and the `\Cref{sec:autograd-note}` autograd-note template.
- Checklist (sec 8): multimodal p(y|x); the MDN head composing softplus, softmax, logsumexp; **natural extension**, framed exactly like `\Cref{sec:autograd-note}`. Reframe the booklet's "reader exercise" language.
- Commit: `ch02: sections 7-8 link/likelihood independence and the MDN pointer`.

### Task 8: Sections 9 to 11, the parallel-axes synthesis, library additions, handoff

File `06-synthesis-library-handoff.tex`. ~2 pp. Header: DEFINED `sec:parallel-axes`, `sec:library-additions`, `sec:handoff`; RESOLVED `\Cref{sec:inductive-biases}`.

- Checklist (sec 9): OWNS the full two-axes synthesis (Chapter 1's `\Cref{sec:inductive-biases}` previews; this completes it). Head and architecture are independent and composable; matched-vs-mismatched analysis; the composability examples (CNN + Poisson, Transformer + Categorical, MLP + heteroscedastic Gaussian, CNN + Bernoulli-per-pixel). Do not merely repeat Chapter 1; deliver the synthesis it deferred here.
- Checklist (sec 10): library additions, PRESENT TENSE (DE-SERIALIZE: the booklet says "to add"; these are already in `scratchnn` on `main`). `softplus`, `MSELoss`, `PoissonNLLLoss`, `GaussianNLLLoss`, each fitting the `Loss` interface, each with a `gradient_check` case. Beta and MDN remain out, as natural extensions.
- Checklist (sec 11): recap, then forward. The architecture axis continues in the NEXT chapter, CNN (prose-only forward reference, since Chapter 3 is not yet drafted; name it by title, do not `\Cref`). Heads compose with any body; RL closes the book.
- Commit: `ch02: sections 9-11 synthesis, library additions, handoff`.

### Task 9: Bibliographic Notes

File `bib-notes.tex`, unnumbered `\section*{Bibliographic Notes}`. ~0.5 pp. Header: RESOLVED `\cite` keys only.

- Checklist: generalized linear models and canonical links (Nelder and Wedderburn 1972; McCullagh and Nelder 1989); heteroscedastic Gaussian regression (Nix and Weigend 1994); mixture density networks (Bishop 1994); reuse MacKay 2003 and Goodfellow et al. as standard references. Add new entries to `book/references.bib`. Confirm each `\cite` resolves under biber.
- Commit: `ch02: bibliographic notes and references.bib entries`.

### Task 10: Integration and verification

- **Build:** `make -C book cleanall && make -C book` runs clean (the cleanall guards against any stale-build edge case); exit 0.
- **Cross-reference audit:** every `\label` defined; every backward `\Cref` to Chapter 1 resolves; `book.log` shows ZERO "Reference undefined" (the Chapter 2 baseline is empty). Run the cross-ref-auditor.
- **Figures placed (the Chapter 1 S1 lesson):** confirm both `fig:ch02-poisson` and `fig:ch02-heteroscedastic` are `\includegraphics`'d and their caption text appears in the built PDF (`pdftotext`), not merely sitting in `figures/`.
- **Real-code audit:** diff `PoissonNLLLoss`, `GaussianNLLLoss`, `softplus` listings against `../../src/scratchnn/neural_net.py`.
- **Math/numbers audit:** the canonical-link theorem and its proof; the Poisson and heteroscedastic numbers reconciled with the notebook.
- **De-serialization audit:** no "post" idiom; CNN is forward-referenced (not "already done"); section 10 is present-tense; Beta/MDN are "natural extensions", not "exercises"; section 9 delivers the synthesis (not a repeat of Chapter 1).
- **Voice / head-overflow:** no em-dashes or banned phrases; distinguish overfull `\hbox` (must be 0 in heads) from benign underfull `\vbox`.
- **Page budget:** ~10 pp.
- Commit: `ch02: integration pass; build clean, figures placed, listings verified`.

## 9. Risks

| Risk | Mitigation |
|---|---|
| The Poisson/heteroscedastic example scripts may not exist; the notebook stalls. | Build both experiments in the notebook from the booklet's specified setups (functions, architectures, seeds). |
| A fresh seed shifts the specific numbers (table, NLL values). | Notebook-first: write prose to the regenerated values. State seeds. |
| A figure is generated but never placed (the Chapter 1 S1 bug). | `\includegraphics` is a checklist item in Tasks 5 and 6; Task 10 verifies the caption is in the PDF. |
| The canonical-link theorem proof balloons past its budget. | Keep the proof short (the A'(eta) = E[T(y)] identity plus one line); the worked instances live in sections 4 and 5, not the proof. |
| Section 9 repeats Chapter 1's section 7 instead of completing it. | Chapter 1 only previews and names the axes; Task 8 must deliver the composability matrix and matched/mismatched analysis that Chapter 1 deferred. |
| Blog-order reference to CNN as "already done" survives. | Explicit de-serialization checklist items in Tasks 1 and 8; Task 10 audits for it. |

## 10. Out of Scope (Chapter 2)

- General GLM theory beyond the single canonical-link theorem (no Fisher scoring, no IRLS).
- Implementations of Beta NLL or the MDN (natural-extension pointers only).
- Any Part II or III content beyond the prose forward references in the handoff.
- Exercises (book has none).

## 11. Success Criteria for Chapter 2

1. A reader can state why `p - y` recurs (the canonical-link theorem) and can choose an output head for real-valued, binary, categorical, count, and heteroscedastic targets, justifying each as a prior.
2. Every code listing matches `src/scratchnn`; both experiment figures are placed and their numbers regenerate from the notebook.
3. Backward `\Cref`s into Chapter 1 all resolve; the forward CNN reference is prose-only; zero undefined refs.
4. No blog-serialization artifacts: CNN is forward-referenced, section 10 is present-tense, Beta/MDN are natural extensions, and section 9 completes (not repeats) the two-axes synthesis.
5. The chapter reads as the closing of the supervised-learning frame and motivates the architecture axis that Part II takes up.

## 12. Next Step

`/bookwright:draft chapter2` to execute this plan (scaffold and notebook first, then prose, then integration).
