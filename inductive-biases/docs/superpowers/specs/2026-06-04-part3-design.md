# Design: Part III (Interpretation and Beyond)

**Author:** Alexander Towell
**Date:** 2026-06-04
**Status:** Design (pre-implementation)
**Parent spec:** `docs/superpowers/specs/2026-06-02-master-design.md`
**Scope:** Section-level outline for Chapters 7 and 8 (Part III, ~23 pp). Authoritative for the Part III implementation plans. The book's final part.

## 1. Purpose

The first two parts ask what bias to build in. Part III asks what the network did with it, and then leaves supervised learning behind. Chapter 7 takes the trained pointer transformer of Chapter 6 apart and reads the bias back out of the weights, the empirical payoff of the book's third axis (implementation realization). It is the book's flagship honest-empiricism result: the small model confirms the architectural prediction cleanly, and the scaled model refutes the obvious story, with a causal probe revealing the truth that the attention maps hide. Chapter 8 changes the learning signal entirely, trading per-example labels for a scalar reward over trajectories, and shows that the supervised toolkit (softmax cross-entropy) is the inner loop of reinforcement learning. Chapter 8 is the last chapter; its closing section is the book's conclusion.

Part III is a re-homing of booklet chapters 7 and 8. The prose largely exists; the work is porting it, regenerating the figures and numbers from paired notebooks, completing the de-serialization (more blog idiom survives here than elsewhere), and writing the book's ending.

## 2. Inherited Commitments

All master-spec and Part I/II conventions: self-study monograph, single voice, no em-dashes (soul hook), no exercises, real code from source, per-chapter Bibliographic Notes to `references.bib`, auto-numbered `\section` with `\Cref` (prose-only for not-yet-drafted forward refs), per-file header blocks, notebook-first with stored outputs, place figures with `\includegraphics` + `\Cref`, regime-reconciliation when a number changes, clean-rebuild to verify.

## 3. Decisions Settled for Part III

**(A) Ch7 figures regenerated as a curated key set.** The paired notebook re-runs the probes and generates the essential figures: `fig:causal-trace` (the flagship M=32 activation-patching trace), `fig:attn-contrast` (a legible M=8 attention map beside an illegible M=32 one), and `fig:lookup-by-address` (the per-head dereference weight by address). The existing series PNGs at `docs/series/figures/07-interp{,-deep}/` are reference, not the source. Other per-address heatmaps stay out (figures where they add most).

**(B) Ch8 closing is a full book-wrap.** Chapter 8 is the last chapter, so `sec:book-close` is the book's conclusion: recap the three axes across all eight chapters, the journey from XOR (Chapter 1) to causal tracing (Chapter 7), and close on AIXI as the incomputable north star. Fix the booklet's stale "seven-post arc" count to eight chapters.

**(C) torch enters the env for Chapter 7.** Chapter 7's M=32 act (§9, the flagship) reverse-engineers a PyTorch model. The notebook loads the committed checkpoint `examples/pytorch/interp_deep_M32.pt` (no training, runs in seconds) and runs the PyTorch probes (causal flip test, activation patching, causal trace). So `pyproject.toml` gains `torch` (a notebook dependency). This is justified: the chapter's subject is a PyTorch model. The M=8 act (§1 to 8) is NumPy and needs no torch.

**(D) Ch7 studies two distinct models; say so.** The M=8 confirmatory analysis (§1 to 8) is the from-scratch NumPy `BitTransformer` (2L, 1H, d=32; `examples/pointer_interp.py`, retrains cheaply each run). The M=32 flagship (§9) is a separately-trained introspectable PyTorch model (3L, 4H, d=128, hand-rolled attention; `examples/pytorch/pointer_interp_deep.py`), NOT the Chapter 6 sweep model (which used `nn.TransformerEncoderLayer`). The spec and prose must state this plainly; accuracies are not comparable across the two.

**(E) De-serialization (heaviest in the book).** The `.tex` files retain blog idiom: Ch7 "post-6"/"Post 6 found" (§4, §9), a garbled §10 ("The the book identified three axes"), the closing "next ... post"; Ch8 "seven-post arc", "seven posts", "a dedicated series is forthcoming", "the language-model post". Convert all to chapter language; fix the seven/eight count; reword the §8 pointer to `docs/series/archive/transformer-text.md` (an archived source, not a chapter). Port from the booklet `.tex`, never the series `.md`.

**(F) Experiments run on NumPy/PyTorch; scratchnn is cited, not run.** Consistent with Chapter 6's boundary. Ch7: NumPy (M=8) + PyTorch (M=32). Ch8: pure NumPy (`examples/reinforce_gridworld.py`, a `PolicyMLP` that deliberately mirrors the `scratchnn` `Linear`/`Tanh`/`SoftmaxCrossEntropy` stack). The §6 bridge cites `scratchnn`'s `SoftmaxCrossEntropy` gradient (p minus the one-hot action) as the conceptual through-line.

## 4. Chapter 7: Reverse-Engineering the Pointer Transformer (~13 pp, has notebook)

Title: **Reverse-Engineering the Pointer Transformer**. Label `\label{ch:interpretability}`. Source: booklet ch7, series 07; code from `examples/pointer_interp.py` (NumPy) and `examples/pytorch/pointer_interp_deep.py` (PyTorch).

| # | Title (`\label`) | pp | Source | Purpose |
|---|---|---|---|---|
| 1 | The setup `sec:interp-setup` | 0.75 | booklet | The plan: read Chapter 6's architectural prediction off the trained weights. |
| 2 | What the circuit must be `sec:circuit-restated` | 0.75 | booklet | Restate the two-stage necessity argument from `\Cref{sec:one-layer-limit}`: aggregate the address, then dereference. |
| 3 | Layer 2 attends to the addressed position `sec:layer2-deref` | 1.25 | NumPy probe; figure | M=8: layer 2's argmax is the address; averaged weight on m_a is 0.66 to 0.81. Part of `fig:attn-contrast`. |
| 4 | Layer 1 is the address aggregator `sec:layer1-aggregator` | 1.0 | NumPy probe | Layer 1 puts 93 to 99 percent of mass on the three address positions, MSB first. |
| 5 | Causal confirmation: ablations `sec:ablations` | 1.25 | NumPy probe | Baseline 1.000; L1 to uniform 0.658; L2 to uniform 0.636; shuffle address 0.776. The prediction is causally confirmed. |
| 6 | Is this an induction head? `sec:induction-head` | 1.0 | booklet | QK/OV reading; relation to induction heads (Olsson) and the circuits framework (Elhage). |
| 7 | ICL is content-addressable lookup, composed `sec:icl` | 0.75 | booklet | In-context learning as composed lookups; the minimal cousin of induction. |
| 8 | What this small model leaves out `sec:small-model-limits` | 0.75 | booklet | Honest limits; reword the archived text-transformer pointer. |
| 9 | Scaling the lens: attention hides the M=32 circuit, causal tracing reveals it `sec:scaling-the-lens` | 4.5 | PyTorch probe; 2 figures | THE FLAGSHIP. The honest arc: two pre-registered hypotheses refuted (layer 2 does not dereference, 0.047; heads cover only 10 of 32 addresses); the causal flip test (0.995 vs 0.005) and causal trace (m_a transported to the readout: embedding 1.000 to L3 readout 1.000) prove a perfect pointer. "Attention weight is not information flow." The Kaiming-vs-Xavier init aside ties back to Chapter 6's PE-scale lesson. Figures: `fig:lookup-by-address`, `fig:causal-trace`. |
| 10 | Closing: the inductive-bias frame, completed `sec:interp-closing` | 1.0 | booklet | The interpretability axis sits across all three (`\Cref{sec:three-axes}`); the meta-lesson (causal probes beat weight-reading). Forward (prose) to Chapter 8. |

### Chapter 7 Notebook (`ch07-interpretability.ipynb`)
- M=8 (NumPy, `pointer_interp.py`): retrain the 2L/1H model (seed-fixed), regenerate the layer-2 dereference weights, the layer-1 address attention (93 to 99 percent), and the ablation table (0.658 / 0.636 / 0.776). Generate the M=8 half of `fig:attn-contrast`.
- M=32 (PyTorch, `pointer_interp_deep.py`): load `interp_deep_M32.pt` (no training); run the causal flip test (0.995 / 0.005), per-layer ablations (0.540 / 0.763 / 0.600), the per-head-by-address weights, and the activation-patching causal trace. Generate `fig:lookup-by-address`, `fig:causal-trace`, and the M=32 half of `fig:attn-contrast`.
- Needs `torch` (decision C). Fix seeds; execute end to end with stored outputs.

## 5. Chapter 8: Reinforcement Learning (~9 pp, has notebook, closes the book)

Title: **Reinforcement Learning: A Scalar Reward over Trajectories**. Label `\label{ch:rl}`. Source: booklet ch8, series 08; code from `examples/reinforce_gridworld.py` (NumPy).

| # | Title (`\label`) | pp | Source | Purpose |
|---|---|---|---|---|
| 1 | The trichotomy `sec:trichotomy` | 0.75 | booklet | Supervised / self-supervised / reinforcement; what changes is the signal. |
| 2 | The reduction loses what makes RL hard `sec:rl-reduction` | 0.75 | booklet | "RL is just supervised on (state, action) pairs" misses temporal credit assignment. |
| 3 | The MDP framing `sec:mdp` | 0.75 | booklet | States, actions, rewards, returns; the formal object. |
| 4 | Credit assignment, three families `sec:credit-assignment` | 0.75 | booklet | Value-based, policy-gradient, model-based. |
| 5 | REINFORCE, derived `sec:reinforce` | 1.0 | booklet | The policy-gradient theorem and the REINFORCE estimator. |
| 6 | The bridge: REINFORCE is weighted cross-entropy `sec:reinforce-bridge` | 1.0 | booklet | REINFORCE is softmax cross-entropy weighted by return; the gradient is the p minus one-hot of `\Cref{sec:softmax}`. The `scratchnn` toolkit is the inner loop of RL. |
| 7 | Worked example: REINFORCE on a 5x5 gridworld `sec:gridworld` | 1.5 | notebook; figure | NumPy gridworld; before training return -0.126 / 24.2 steps / 11.5 percent success; after 2000 episodes +0.928 / 8.21 steps / 100 percent. Figure: `fig:gridworld-learning-curve`. |
| 8 | AIXI: the theoretical north star `sec:aixi` | 0.75 | booklet | AIXI as Solomonoff induction (`\Cref{ch:fixed-context-lm}`) plus Bayesian decision theory plus reward maximization. |
| 9 | Inductive bias, the RL axis `sec:rl-axis` | 1.0 | booklet | Reward shaping, policy architecture (the Part II families re-cited), exploration as priors; the heads-as-bias frame (policy = softmax head, value = identity plus MSE head). |
| 10 | Closing: the book, completed `sec:book-close` | 1.0 | booklet, expanded | THE BOOK'S CONCLUSION (decision B). Recap the three axes across all eight chapters; the arc from XOR to causal tracing; AIXI as the north star, REINFORCE its smallest computable shadow. Fix "seven" to eight. No forward references. |

### Chapter 8 Notebook (`ch08-rl.ipynb`)
Pure-NumPy gridworld REINFORCE (`reinforce_gridworld.py`): seed-fixed run of 2000 episodes (gamma 0.99, lr 0.05); regenerate the before/after numbers, the success rate, the trained greedy path, and the smoothed learning curve. **Figure:** `fig:gridworld-learning-curve` (mean return vs episode). No torch.

## 6. Forward and Backward Reference Map

Part III chapters are drafted in order (7, then 8). All cross-chapter references are backward (to Parts I and II) and resolve, except Ch7's single forward reference to Chapter 8.

| From | Direction | To | Mechanism |
|---|---|---|---|
| `ch:interpretability` §sec:circuit-restated | backward | `\Cref{sec:one-layer-limit}`, `\Cref{ch:transformer}` | resolves |
| `ch:interpretability` §sec:scaling-the-lens | backward | Chapter 6's PE-scale lesson (`\Cref{sec:pe-scale-fix}`) | resolves |
| `ch:interpretability` §sec:interp-closing | backward / forward | `\Cref{sec:three-axes}` (backward); Chapter 8 (forward, prose-only until drafted) | mixed |
| `ch:rl` §sec:reinforce-bridge, §sec:rl-axis | backward | `\Cref{sec:softmax}`, `\Cref{sec:parallel-axes}`, the architecture axis `\Cref{ch:cnn}`/`\Cref{ch:rnn}`/`\Cref{ch:transformer}` | resolves |
| `ch:rl` §sec:aixi | backward | `\Cref{ch:fixed-context-lm}` (the Solomonoff north star) | resolves |
| `ch:rl` §sec:book-close | backward | all parts (`\Cref{ch:foundations}`, `\Cref{sec:inductive-biases}`, `\Cref{sec:three-axes}`, ...) | resolves |

Chapter 8 makes no forward references (it is the last chapter). Note: Chapter 8 does not reference Chapter 7 (it ties back to the supervised arc); that is intentional, not a gap.

**Expected-unresolved baseline per chapter at draft time: EMPTY.** Ch7's forward reference to Chapter 8 is prose-only until Chapter 8 is drafted.

## 7. Page Budget

| Block | Pages |
|---|---|
| Part III divider blurb | ~1 |
| Ch 7 Interpretability | ~13 |
| Ch 8 Reinforcement Learning | ~9 |
| **Total** | **~23** |

Consistent with the master spec's Part III target. Ch7's §9 is the largest single section in the book (~4.5 pp) and is allowed to be; its figures and tables carry the argument.

## 8. Sequencing for the Implementation Plans

Draft Chapter 7 first (it builds directly on Chapter 6's trained model and `sec:three-axes`), then Chapter 8 (which closes the book). Each runs notebook-first: execute the paired notebook (Ch7 with torch; Ch8 NumPy), fix seeds, store outputs, generate the decided figures, then write the prose to the regenerated numbers, wire `\label`s and `\Cref`s, apply the de-serialization pass, audit, integrate.

Ch7's plan must add `torch` to `pyproject.toml`, confirm `interp_deep_M32.pt` loads, and verify the two-models framing (decision D). Ch8's plan must write the book-wrap (decision B) and the RL bibliographic notes (Williams 1992 REINFORCE; Sutton and Barto; Silver et al. AlphaZero; Hutter for AIXI, already in `references.bib`).

## 9. Open Action Items

1. **torch in the env (Ch7).** Declare `torch` in `pyproject.toml` (a notebook dependency or extra); verify `interp_deep_M32.pt` loads and the probes run. The M=8 cells are NumPy.
2. **Ch7 figures into the book tree.** The notebook generates `fig:causal-trace`, `fig:attn-contrast`, `fig:lookup-by-address` into `book/figures/`; sections place them (S1 lesson). The existing series PNGs are reference only.
3. **De-serialization targets (decision E).** Ch7 "post-6"/"Post 6", the garbled §10, the archive pointer; Ch8 "seven-post arc"/"seven posts"/"dedicated series forthcoming"; the seven-to-eight count. Auditor checks at integration.
4. **Ch7 two-models clarity (decision D).** State that §1 to 8 is the NumPy M=8 model and §9 is a fresh introspectable PyTorch M=32 model, not the Chapter 6 sweep model; accuracies are not comparable.
5. **RL bibliographic entries (Ch8).** Add `williams1992reinforce`, `sutton2018rl`, `silver2017alphazero` (and confirm `hutter2005uai`) to `references.bib`.
6. **Book-wrap (Ch8 §10).** Write the conclusion to the whole book (decision B), not just the RL chapter.

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Adding torch to the env is heavy or breaks the build. | torch is a notebook-only dependency (the book build does not need it); pin a CPU build; the checkpoint loads in seconds, no training. |
| The Ch7 M=8 retrain shifts the probe numbers on a fresh seed. | Notebook-first; fix the seed; reconcile the prose probe numbers to the run. The qualitative claims (layer 1 aggregates, layer 2 dereferences, ablations break it) are robust; the exact weights may move a little. |
| The flagship §9 numbers (causal trace, ablations) come from the checkpoint and must match the prose exactly. | The checkpoint is fixed, so the M=32 numbers are reproducible; quote them from the notebook run. |
| The book-wrap over-reaches or repeats. | Recap the three axes and the arc concisely; do not re-explain each chapter. The close is a landing, not a summary lecture. |
| The heavy de-serialization misses a leftover (this part has the most). | The reference map and decision E are the checklist; the integration auditor greps for "post", "series", "seven". |

## 11. Out of Scope (Part III)

- A full interpretability methods survey (Chapter 7 uses these specific probes on this specific model).
- A full RL course (Chapter 8 derives REINFORCE and runs one gridworld; Sutton and Barto remains the depth reference).
- Deep RL at scale, RLHF mechanics, world models (named in the close as the blur, not developed).
- A separate conclusion chapter (Chapter 8 §10 is the book's close).
- Exercises.

## 12. Success Criteria for Part III

1. A reader sees the architectural prediction confirmed on the small model and then sees the obvious story refuted and corrected by a causal probe on the scaled model, and understands the meta-lesson (attention weight is not information flow).
2. Chapter 8 shows RL's distinctive hard part (credit assignment) and that REINFORCE is the supervised softmax cross-entropy of `\Cref{sec:softmax}` weighted by return; the gridworld numbers regenerate from the notebook.
3. Every figure is generated by the paired notebook and placed; Ch7's notebook runs the M=8 NumPy and M=32 PyTorch probes with stored outputs; Ch8's gridworld runs with stored outputs.
4. The two-models framing is explicit; backward `\Cref`s into Parts I and II all resolve; zero undefined refs.
5. The blog idiom is gone (no "post", "series", "seven-post arc"); the chapter count is correct.
6. The book ends well: Chapter 8 §10 closes the whole book on the three axes and AIXI, a real ending rather than a quiet stop.

## 13. Next Step

`/bookwright:plan chapter7` to write the Chapter 7 (interpretability) implementation plan, notebook-first, adding torch for the M=32 probes. Then `/bookwright:plan chapter8` for the closing chapter.
