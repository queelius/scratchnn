# Master Design Spec: Inductive Biases in Neural Networks

**Author:** Alexander Towell
**Date:** 2026-06-02
**Status:** Design (pre-implementation)
**Scope:** Book-level design. Authoritative for all per-Part design specs and per-chapter implementation plans.
**Project:** `inductive-biases/` (bookwright), nested in the `scratchnn` repository.
**Subtitle:** A Built-from-Scratch Tour

## 1. Purpose and Thesis

The book is about inductive bias: the assumptions a learning system brings to the data before it sees any. A model that assumes nothing learns nothing, so every useful network is a particular bundle of priors. The book takes that idea apart and rebuilds the major neural architectures around it.

Three axes organize the whole book, and every chapter is located explicitly on them:

1. **Output head.** A loss function encodes a prior over the distribution of the response. Choosing it (Bernoulli, Categorical, Gaussian, Poisson) is the first modeling decision. This is the first inductive bias the reader meets, in Chapter 2.
2. **Architecture.** The wiring of a network is a prior over how features compose. Locality, recurrence, and content-addressable lookup are three such priors made concrete (Chapters 3 to 6).
3. **Implementation realization.** Once a model is trained, a third axis appears: whether gradient descent actually finds the solution the architecture permits, and whether we can read that solution back out of the weights (Chapter 7).

The book is a re-homing of an existing, complete artifact. The eight chapters began as a blog series (`../../docs/series`) and were assembled into a single-file LaTeX booklet (`../../docs/booklet`) that already builds to an 86-page PDF. This spec does not invent structure. It codifies the structure that exists, elevates the chapters to a coherent monograph through the bookwright pipeline (cross-reference threads, per-chapter bibliographic notes, paired sanity-check notebooks, voice and code-listing discipline), and adds nothing the posts do not already promise.

## 2. Audience and Prerequisites

**Primary reader: the self-study practitioner.** Someone who knows some machine learning, can read about 300 lines of straightforward Python, and wants to understand *why* the standard architectures work rather than only how to call them. This is the broadest band and it matches the standalone-posts origin of the material.

**Assumed prerequisites:**

- Single-variable and partial derivatives, and the chain rule. The book hand-derives every gradient; the reader must be able to follow that.
- Linear algebra at the level of vectors, dot products, and matrix-vector products. No eigendecomposition or spectral theory is assumed.
- Basic probability: random variables, expectation, and the Bernoulli, Categorical, and Gaussian distributions by name.
- Python literacy sufficient to read (not necessarily write) clear, dependency-free code.

**Explicitly not assumed:** measure theory, functional analysis, prior experience with PyTorch or any deep-learning framework, or a background in information theory. Where a chapter needs more (the BSC, capacity, attention), it builds the needed idea in place.

**Consequence of the audience choice (declined alternatives matter):** the reader picked self-study over course-adoption, and *no exercises* (Section 7). The book therefore stays a readable monograph. It does **not** carry formal learning-outcome boxes, problem sets, or a solutions appendix. Active learning is carried by the paired notebooks and by in-text worked examples.

## 3. Format, Tone, and Voice

A pedagogical monograph in the MacKay tradition: a single authorial voice, intuition first, then hand-derived math, then real numerical verification. Each chapter reads as follows.

1. **Motivate.** Name the prior the chapter is about, in plain language.
2. **Derive.** Work the math by hand. The chain rule is local and visible, never one big opaque expression.
3. **Implement.** Show the *real* `scratchnn` code that realizes the idea (Section 6).
4. **Test.** Run the smallest model that isolates the prior and report what happened, including when the tidy hypothesis was refuted.
5. **Reflect.** Locate the chapter on the three axes and point forward.

**Voice rules (soul plugin, enforced on `.tex` writes):** no em-dashes. Use commas, colons, periods, or parentheses instead. Single voice throughout. This spec is written to the same rule so it models the book.

**Honest empiricism is a feature, not a hedge.** Where an experiment refuted a tidy story, the chapter keeps the messier truth. The flagship instance is Chapter 7, where the obvious "read it off the attention map" story is wrong and a causal probe is needed to find the right one. This honesty is a stated success criterion (Section 14).

## 4. Structure

Three parts, eight chapters. Titles are taken from the booklet table of contents (the most curated form) and may be refined during per-Part design. "Axis" locates the chapter on the three organizing axes of Section 1. "NB" marks chapters that get a paired notebook (Section 7). Pages are current booklet span then target (Section 10).

### Part I: Foundations (the floor before the bias)

| Ch | Title | Axis | Pages | NB | Primary source |
|---|---|---|---|---|---|
| 1 | Foundations: Function Approximation and the Multilayer Perceptron | Pre-axis: capacity | 10 to 11 | yes | series 01, booklet ch1, `neural_net.py` |
| 2 | Output Heads as Inductive Bias | Axis 1: output head | 9 to 10 | yes | series 02, booklet ch2, `neural_net.py` losses |

Part I builds the floor. Chapter 1 starts from unconstrained function approximation, shows why one linear unit cannot separate XOR, and recovers the MLP as the smallest fix. Chapter 2 introduces the first inductive bias proper: the output head. The same network body becomes a classifier, a count model, or a regressor depending only on the head bolted on. The two axes that organize the rest of the book (architecture and output head) both first appear here.

### Part II: Architectural Inductive Biases (a prior over how features compose)

| Ch | Title | Axis | Pages | NB | Primary source |
|---|---|---|---|---|---|
| 3 | Convolutional Networks: Locality and Translation Equivariance | Axis 2: architecture | 7 to 8 | yes | series 03, booklet ch3, `conv.py` |
| 4 | Fixed-Context Language Models | Axis 2: architecture | 6 to 7 | yes | series 04, booklet ch4, `embedding.py` |
| 5 | Recurrent Networks: Time-Translation Equivariance | Axis 2: architecture | 7 to 8 | yes | series 05, booklet ch5, `recurrent.py` |
| 6 | Attention as Content-Addressable Memory | Axis 2: architecture | 11 to 12 | yes | series 06, booklet ch6, NumPy/PyTorch |

Each chapter states the structural prior, builds a model small enough to test it, and checks whether the assumption holds. CNNs assume locality and translation equivariance. Fixed-context LMs assume a bounded window suffices (Bengio 2003). RNNs assume time-translation equivariance and squeeze the past through a state bottleneck. Transformers assume nothing about which positions matter and instead learn a content-addressable lookup, position by position. This is where pure Python runs out of budget: Chapter 6 moves to NumPy and PyTorch and says so.

### Part III: Interpretation and Beyond

| Ch | Title | Axis | Pages | NB | Primary source |
|---|---|---|---|---|---|
| 7 | Reverse-Engineering the Pointer Transformer | Axis 3: realization | 13 to 14 | yes | series 07, booklet ch7, PyTorch + probes |
| 8 | Reinforcement Learning: A Scalar Reward over Trajectories | Beyond supervised | 10 to 11 | yes | series 08, booklet ch8, RL experiment |

The first two parts ask what bias to build in. Part III asks what the network did with it. Chapter 7 takes a trained transformer apart and finds that the right architecture is necessary but not sufficient: implementation realization governs whether gradient descent finds the bias-permitted solution and whether we can read it back. The lesson is cautionary about staring at attention maps and is the book's most useful result. Chapter 8 changes the learning signal entirely, trading per-example labels for a scalar reward over whole trajectories, and asks how much of the inductive-bias frame survives the move to reinforcement learning.

## 5. Running Threads

Five cross-cutting commitments. The per-Part specs and per-chapter plans must verify each thread carries from the prior chapter.

| Thread | Introduced | Carries through | Pays off |
|---|---|---|---|
| The three axes (head, architecture, realization) | Preface, Ch 1 to 2 | Every chapter names its axis | Ch 7 (axis 3 made explicit), Ch 8 (frame stress-tested) |
| The `p - y` gradient | Ch 2 (notation note flags it early) | Recurs at every canonical loss | Ch 2 reveals binary logistic is two-class softmax with one logit pinned to zero |
| Scientific rhythm: state prior, build smallest model, test it | Ch 1 | Every Part II chapter | Ch 3 to 6 each report whether the assumption held |
| `scratchnn` as the through-line (real, inlined code) | Ch 1 | Library grows: core, conv, embedding, recurrent | Ch 6 to 8 switch to NumPy/PyTorch and say why |
| Honest empiricism (refuted hypotheses kept) | Preface | Wherever an experiment surprised | Ch 7 flagship: the obvious story is wrong |

The `p - y` thread is already seeded in the booklet notation front matter ("the gradient of every canonical loss with respect to its logits is `p - y`; when you see it again, that is not a coincidence"). The per-chapter plans must keep that promise.

## 6. The scratchnn Through-Line and the Real-Code Policy

The book is built on `scratchnn` (`../../src/scratchnn`), a pure-Python neural network library written to be read. The organizing idea of the library is the organizing idea of Chapter 2: **a network emits raw logits, and a loss interprets them.**

**Real-code policy (standing project rule):** every code listing is taken from the actual `scratchnn` source it describes, lightly trimmed. Initialization boilerplate may be elided; the hand-derived `backward` is kept in full, because that is the teachable part. Listings are never pseudocode and never usage-only snippets that hide the mechanism. This rule is load-bearing for the book's pedagogy and is recorded in project memory.

**What the library implements today (verified against `src/scratchnn/__init__.py` on 2026-06-02):**

- Core (`neural_net.py`): `Linear`, `Tanh`, `ReLU`, `Network`, `gradient_check`; the `sigmoid`/`logsumexp`/`softmax`/`softplus`/`dot` helpers. Plain mini-batch SGD via a generic `step`. No autograd, no `Optimizer` class, no NumPy.
- Output heads (all five Chapter 2 discusses, all hand-derived, all fitting the `Loss` interface): `MSELoss` (Gaussian, fixed variance), `SigmoidBCE` (Bernoulli), `SoftmaxCrossEntropy` (Categorical), `PoissonNLLLoss` (Poisson), `GaussianNLLLoss` (heteroscedastic Gaussian). Beta NLL and the mixture density network are intentionally absent: Chapter 2 frames them as natural extensions, the way Chapter 1 frames autograd.
- `conv.py` (Chapter 3), `embedding.py` (Chapter 4), `recurrent.py` (Chapter 5).
- `visualize.py`: the one module permitted third-party dependencies (matplotlib, Pillow), hooked in only through `Network.fit(..., callback=...)`.

**Where pure Python ends.** Chapters 6, 7, and the Chapter 8 RL experiment use NumPy and PyTorch. The book states the switch each time. This is consistent with the preface and is not a defect to hide.

## 7. Notebook Discipline

Per the project notebook stack (`python-uv-jupyter`, configured in `bookwright.config.yaml`), each chapter that makes live numerical claims gets one paired, executed notebook under `notebooks/chNN-slug.ipynb`. The notebook regenerates that chapter's figures and sanity-checks every number the prose asserts.

**Applied to this book, all eight chapters qualify**, because every chapter makes a numerical claim a notebook can regenerate (training curves, gradient-check residuals, accuracy or perplexity figures, the pointer-task scaling study, the causal-probe results, the RL learning curve). The "where it adds value" rule is therefore the standing policy for any future pure-conceptual chapter, not a license to skip notebooks here.

**Tooling reality:**

- Chapters 1 to 5 import `scratchnn` (editable install: `uv pip install -e ../..`). Chapter 1 also exercises `visualize.py`.
- Chapters 6 to 8 need NumPy and PyTorch in the uv environment. `pyproject.toml` must declare these (a notebook extra or dev group). This is an open action item (Section 11).

**No exercises.** The reader declined exercises. Notebooks plus in-text worked examples are the entire active-learning surface. There is no solutions appendix.

## 8. Citation Policy

Per-chapter **Bibliographic Notes** at the end of each chapter, in the monograph tradition. Underlying papers are cited there (for example Bengio 2003 for fixed-context LMs, the attention and transformer lineage for Chapter 6, the relevant interpretability and RL references for Chapters 7 and 8). Light inline citation is allowed where a specific named result is invoked in the prose.

The single bibliography is `book/references.bib` (biblatex with the biber backend, per `book.tex` and the project Makefile). The booklet's older `references.tex` is a source to port, not the build target. During integration, every work cited in a Bibliographic Notes block must resolve to a `references.bib` entry.

## 9. Repository Layout and Source Material

The bookwright project tree (already scaffolded and committed) is the build target:

```
inductive-biases/
  book/            book.tex, preamble.tex, alex.sty, references.bib, Makefile
    parts/         part1.tex, part2.tex, part3.tex (one \part each)
    chapters/      01-... .tex through 08-... .tex (authored here)
    frontmatter/   preface, notation
  notebooks/       ch01-... .ipynb through ch08-... .ipynb (paired, executed)
  docs/superpowers/specs/   this file, then per-Part design specs
  docs/superpowers/plans/   per-chapter implementation plans
```

**Source material (read-only inputs, in the parent repo):**

- `../../src/scratchnn`: the library. The authority for every code listing.
- `../../docs/series`: the eight blog posts. The prose seed.
- `../../docs/booklet`: the existing single-file LaTeX assembly (chapters, preface, notation, part blurbs, references). The closest thing to a first draft of the book; port and refine from here.

The bookwright `book/parts/part1.tex` currently holds only a `\part{Foundations}` stub. The booklet's three part blurbs (already written, one paragraph each) are the canonical part introductions to port.

## 10. Page Budget

The current booklet is 86 pages. The re-home target is approximately **95 pages**. The uplift is apparatus (per-chapter Bibliographic Notes, a small number of notebook-regenerated figures, thread wiring), not new prose, consistent with the no-new-chapters, no-exercises scope.

| Block | Pages |
|---|---|
| Front matter (title, TOC, preface, notation) | ~6 |
| Part I (Ch 1 to 2) | ~22 |
| Part II (Ch 3 to 6) | ~34 |
| Part III (Ch 7 to 8) | ~26 |
| Back matter (references, short further-reading) | ~7 |
| **Total** | **~95** |

Per-chapter targets are in the Section 4 tables (current span then target). If a chapter overshoots its target during drafting, the first lever is to push a numerical worked example into the paired notebook rather than expand the prose.

## 11. Sequencing and Build Plan

Because the content already exists and is sequential, the natural port order is reading order. The threads accrue front to back, and the `scratchnn` library grows in the same order (core, then conv, embedding, recurrent, then the NumPy/PyTorch chapters). Unlike a greenfield textbook, there is no need to draft a later chapter first to lock definitions.

**Per-Part bookwright workflow:**

1. `/bookwright:design partN` writes the section-level spec for the part (porting the booklet chapter structure into section tables, in the manner of the reference template).
2. `/bookwright:plan chapterN` writes the per-chapter implementation plan.
3. `/bookwright:draft` ports and refines the prose from booklet plus series, verifying every code listing against current `scratchnn`.
4. `/bookwright:notebook chN` drafts and executes the paired notebook.
5. `/bookwright:review` runs the multi-agent editorial review.
6. `/bookwright:integrate partN` writes the integration-pass record.

**Recommended order:** Part I, then Part II, then Part III. Within each part, draft in chapter order.

**Drafting is port-and-refine, not greenfield.** The verification focus for every chapter is: (a) code listings match current `scratchnn` source, (b) every number is regenerated by the paired notebook, (c) the five running threads are wired (Section 5), (d) voice and soul compliance (no em-dashes, single voice), (e) Bibliographic Notes resolve to `references.bib`.

## 12. Open Action Items

Deferred for resolution during per-Part design or drafting.

1. **Resolved (2026-06-02): output heads.** This item assumed `scratchnn` implemented only two heads, citing the older `docs/design.md`. The live library implements all five Chapter 2 discusses (`MSELoss`, `SigmoidBCE`, `SoftmaxCrossEntropy`, `PoissonNLLLoss`, `GaussianNLLLoss`), each with real hand-derived code. Beta NLL and the MDN are deliberate natural-extension pointers. There is no gap to close; keep listings in sync with source at integration. See the Part I spec, Section 4.
2. **Notebook dependencies (Part II / III).** `pyproject.toml` must declare NumPy and PyTorch for the Chapter 6 to 8 notebooks. Pin versions and decide whether they live in the base dependencies or a notebook extra.
3. **Chapter title harmonization.** The series titles and booklet titles differ slightly (for example Chapter 1 includes "Classification" in the series, not in the booklet). Pick the canonical title per chapter during per-Part design; the booklet titles are the default.
4. **Section numbering convention.** The booklet notation note says section numbers are authored (part of each section's title) so they read the same as the standalone posts. Confirm this convention survives into the bookwright build, or switch to LaTeX `\section` auto-numbering. Resolve in Part I design.
5. **Figure provenance.** Existing figures live under `../../docs/series/figures`. Decide whether the paired notebooks regenerate them into `book/` or whether some static figures are copied as-is. Prefer regeneration for any figure backing a numerical claim.

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Re-homing drifts from the coherent booklet and silently changes the argument. | Port from `docs/booklet` as the first draft. Per-chapter plans diff against the booklet chapter and flag any intended change. |
| Code listings rot against the live `scratchnn` source. | The real-code rule is a verification step in every chapter plan: listings are pulled from source and checked at integration. |
| The `p - y` thread is asserted in Chapter 2 but not actually carried. | The thread table (Section 5) is the source of truth; each plan includes "thread carries from prior chapter" as a checklist item. |
| Chapters 6 to 8 notebooks fail to execute in CI or on a fresh env (torch, seeds, runtime). | Pin dependencies (action item 2), fix seeds, and keep the pointer-task and RL runs small enough to execute end to end in the notebook. |
| Page budget creeps past ~95 pp through expanded prose. | No new chapters, no exercises. Overflow goes to the notebook, not the page. |
| The four-heads promise (action item 1) is left ambiguous and Chapter 2 ships with a gap between prose and code. | Force the decision in Part I design before Chapter 2 drafting. |

## 14. Out of Scope

The book does not become, and a drift toward any of these is a signal to trim rather than expand:

- A general deep-learning textbook. It is organized around inductive bias specifically, not around coverage of every architecture or training trick.
- A PyTorch tutorial. PyTorch appears only where pure Python runs out of budget (Chapters 6 to 8) and only in service of the inductive-bias argument.
- A problem book. No exercises, no solutions appendix (reader's decision).
- An autograd or optimization treatise. `scratchnn` has hand-derived per-layer backward and plain SGD by design; autograd is mentioned as the natural generalization, not built.
- New chapters or parts. Scope is re-home and elevate the existing eight chapters.

## 15. Success Criteria

The book is successful if:

1. A self-study reader with undergraduate calculus, linear algebra, and probability can read it front to back and come away able to (a) name the inductive bias a given architecture bakes in, (b) locate any model on the three axes, and (c) explain why the right architecture can still fail to be realized by gradient descent (the Chapter 7 lesson).
2. Every code listing in the book is real `scratchnn` (or, for Chapters 6 to 8, real NumPy/PyTorch) and runs. A reader who opens the repository finds the exact code behind every figure and every number.
3. Every numerical claim is regenerated by a paired notebook that executes end to end.
4. The five running threads feel carried, not contrived. A reviewer can point at any thread's return and say it is doing real pedagogical work.
5. The honest-empiricism commitment holds: at least the Chapter 7 result is presented as a refuted-then-corrected story, and no chapter papers over a hypothesis the experiments did not support.
6. The book reads as one monograph in a single voice, not as eight stitched-together posts, while each chapter still stands alone for a reader who jumps in.

## 16. Next Step

`/bookwright:design part1` to write the section-level design spec for Part I (Chapters 1 and 2), porting the booklet structure into section tables and resolving the Part I open action items (output heads, section numbering). Then `/bookwright:plan chapter1` for the first chapter's implementation plan.
