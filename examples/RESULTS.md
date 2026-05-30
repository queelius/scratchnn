# Pointer-dereferencing experiments: results log

Running record of every experiment on the memory+pointer task
(`simple_pointer_dgp.py`, variant 1). The task: given $M$ memory bits
followed by $A = \lceil \log_2 M \rceil$ address bits, predict the
memory bit at the addressed position. Chance is 0.5; a correct
content-addressable lookup is 1.0.

Two implementations:
- **NumPy** (`examples/*.py`): from-scratch backprop, the pedagogical
  anchor for posts 6 and 7. All math visible.
- **PyTorch** (`examples/pytorch/*.py`): scaling follow-up only. Uses
  `nn.TransformerEncoderLayer` (PreLN, causal mask). Different init
  and internals from the NumPy code, so **do not compare accuracy
  across frameworks**; compare only within a framework.

Common recipe unless noted: Adam, batch 32, 20000 training examples,
2000 held-out test examples.

## M = 8, A = 3 (post 6 §4, post 7)

| Model | Impl | Layers | Heads | $d_{\text{model}}$ | Iters | Test acc |
|---|---|---:|---:|---:|---:|---:|
| MLP baseline (hidden 128) | NumPy | (n/a) | (n/a) | (n/a) | 2000 | 1.000 |
| Transformer | NumPy | 1 | 1 | 32 | 2000 | 0.654 |
| Transformer | NumPy | 2 | 1 | 32 | 2000 | 1.000 |
| Transformer (interp run) | NumPy | 2 | 1 | 32 | 2000 | 1.000 |

The 1L result is the architectural-limit demo: one layer cannot do
dynamic addressing (post 6 §3). The interp run (post 7) reverse-
engineers the 2L1H model: layer 1 aggregates address bits (93-99% of
position-10 attention on the address positions), layer 2 dereferences
(0.66-0.81 weight on the addressed memory cell).

## M = 16, A = 4 (post 6 §10)

| Recipe | Impl | Layers | Heads | $d_{\text{model}}$ | PE | Iters | Test acc |
|---|---|---:|---:|---:|---|---:|---:|
| baseline | NumPy | 2 | 1 | 64 | sinusoidal | 6000 | 0.747 |
| learned PE | NumPy | 2 | 1 | 64 | learned | 6000 | **1.000** |
| kitchen-sink | NumPy | 2 | 4 | 128 | learned | 8000 | **1.000** |

Learned PE alone solves M=16. The PE-scale mismatch (embedding init
0.02 vs sinusoidal scale ~1) was the bottleneck (post 6 §8-9).

## M = 24, A = 5 (post 6 §10)

| Recipe | Impl | Layers | Heads | $d_{\text{model}}$ | PE | Iters | Test acc |
|---|---|---:|---:|---:|---|---:|---:|
| baseline | NumPy | 2 | 1 | 64 | sinusoidal | 6000 | 0.687 |
| learned PE | NumPy | 2 | 1 | 64 | learned | 6000 | 0.605 |
| kitchen-sink | NumPy | 2 | 4 | 128 | learned | 8000 | **0.9975** |

Learned PE alone is *not* enough at M=24 (necessary but not
sufficient). The full kitchen-sink recipe (multi-head + wider +
warmup + learned PE) phase-transitions around iter 7000 and nearly
solves it.

## M = 32, A = 5 (post 6 §10)

| Recipe | Impl | Layers | Heads | $d_{\text{model}}$ | Params | Iters | Test acc |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline (sinusoidal) | NumPy | 2 | 1 | 64 | ~35k | 6000 | 0.664 |
| learned PE | NumPy | 2 | 1 | 64 | ~35k | 6000 | 0.645 |
| kitchen-sink | NumPy | 2 | 4 | 128 | 270k | 8000 | 0.680 |
| kitchen-sink long | NumPy | 2 | 4 | 128 | 270k | 30000 | 0.7625 |
| baseline-match | PyTorch | 2 | 4 | 128 | 270k | 15000 | 0.5950 |
| 2x-width | PyTorch | 2 | 8 | 256 | 1.07M | 15000 | 0.5645 |
| **3L-depth** | PyTorch | **3** | 4 | 128 | 403k | 15000 | **0.7800** |
| wide+deep | PyTorch | 3 | 8 | 256 | 1.59M | 15000 | 0.6225 |

**Key finding: depth is the M=32 knob, not width.** Holding
$d_{\text{model}} = 128$ and heads $= 4$ fixed, going from 2 to 3
layers lifts accuracy 0.595 to 0.780 with a clean phase transition
(loss starts dropping at iter ~7000). Doubling width at 2 layers
(2x-width, 4x the params) does nothing: it stays at chance-plus.

### M=32 extended depth comparison (clean: only layer count varies)

To remove the param-count and convergence-speed confounds from the
sweep above, this run holds $d_{\text{model}} = 128$, heads $= 4$,
$d_{\text{ff}} = 256$, seed, and data fixed, and varies *only*
$n_{\text{layers}}$, at a 40000-iter budget
(`pointer_depth.py`).

| Config | Layers | Params | Iters | Loss | Test acc | Behavior |
|---|---:|---:|---:|---:|---:|---|
| 2L control | 2 | 270k | 40000 | 0.6463 | 0.6160 | flat from iter 2000 on; never transitions |
| 3L treatment | 3 | 403k | 40000 | 0.1154 | **0.9460** | transitions at iter ~7000; still descending |

This is decisive. At 40000 iterations the 2-layer model is still at
chance-plus (0.616) with a loss curve that is essentially flat the
entire run: **time alone does not rescue it**. Adding one layer, with
everything else identical, transitions the model and drives it to
0.946 and still climbing (loss was dropping ~0.01 per 1000 iters at
the end, so a longer budget would likely fully solve it).

**The nuance worth recording.** It is tempting to say "depth equals
the number of composed lookups, and $A = 5$ address bits need a third
layer." But that cannot be the whole story, because M=24 *also* has
$A = 5$ (since $\lceil \log_2 24 \rceil = 5$), and the 2-layer
kitchen-sink solved M=24 to 0.9975. So 2 layers suffice for $A = 5$
address aggregation. What changed between M=24 and M=32 is the
**lookup fan-out** (the number of memory positions to discriminate
among, 24 vs 32) and the fact that M=32 fills the entire 5-bit
address space while M=24 uses only 24 of the 32 codes. The bottleneck
at M=32 is the *precision of the 1-of-M selection*, not the address
decode. The clean empirical claim ("at M=32, depth solves, width and
time do not") stands; the tidy "depth = $\lceil \log M \rceil$
lookups" law does not, and we should not claim it.

## M = 32 mechanism (interp on a solving 3L model, post 7)

`pointer_interp_deep.py` trains an explicit, introspectable 3-layer
model (d_model=128, 4 heads, learned PE) to 0.9965 on M=32 and
dissects it. Two prior hypotheses were **refuted** by the data:
"layer 3 is a refinement stage after a layer-2 lookup" (wrong), and
"the lookup heads partition the address space" (only 10 of 32
addresses are covered by any clean single-head spike).

Implementation-realization note: the explicit model only trains if
the QKV projections use `xavier_uniform_` init. With the default
Kaiming-uniform ($a = \sqrt 5$) init they are mis-scaled and the phase
transition is pushed from iter ~7000 out past iter ~22000, leaving the
model stuck at 0.62 at a 60k budget. Same lesson as the PE-scale
finding, on a different tensor.

What the dissection actually found:

| Probe | Result | Reading |
|---|---|---|
| Per-layer last-position attention | L1 0.51 / L2 0.67 on address positions; L3 holds the dereference | layers 1-2 aggregate the address (two stages), layer 3 dereferences |
| Per-(layer,head) weight on m_a | max is L3H0 = 0.198 (6.3x uniform); other L3 heads near 0 | one partial lookup head, not a clean 1-of-32 pointer |
| L3H0 weight on m_a, by address | spikes 0.8-1.0 for a in {5,6,10,13,17,19}; ~0 elsewhere | the legible lookup covers only a sparse subset of addresses |
| Single-head clean coverage (>0.5) | 10 of 32 addresses | most of the address space has no clean attention spike |
| Layer-3 ensemble weight on m_a | mean 0.29; > 0.5 for only 8 of 32 | even the head ensemble does not visibly concentrate on m_a |
| Per-layer ablation (attn -> uniform) | L1 -> 0.54, L2 -> 0.76, L3 -> 0.60 | layers 1 and 3 essential, layer 2 auxiliary |
| **Causal flip test** | flip m_a -> output tracks it 0.995; flip other cell -> output changes 0.005 | **the model computes exactly m_a, airtight** |
| **Causal trace** (patch clean residual into m_a-flipped run) | recovery at addressed cell 1.00 -> 0.33 -> 0.19 -> 0.00 across embed/L1/L2/L3; readout pos 0.00 -> 0.11 -> 0.30 -> 1.00; other memory ~0 throughout | **m_a is transported cell -> readout over 3 layers**; clean mechanism, invisible to attention, visible to tracing |

The headline: the M=32 model is a *causally* perfect pointer
(flip-m_a tracking 0.995, flip-other 0.005). The clean single-head
attention circuit of the M=8 model (post 7) does not survive the
scale-up, and at first that reads as lost legibility: no single head
or layer-3 ensemble visibly concentrates on m_a for most addresses.
But that was an artifact of the *tool*. A causal trace (activation
patching on minimal pairs) recovers a clean mechanism the attention
maps could not show: m_a is *transported* from the addressed cell to
the readout position over the three layers (addressed-cell recovery
decays 1.00 -> 0 while readout recovery rises 0 -> 1.00; the address
bits are a partial conduit; every other memory cell carries nothing).
The lesson is that the tool, not the model, decides what is legible:
attention weight is suggestive, not probative, and once a computation
spreads across layers and the value pathway you need causal probes
(activation/path patching), not attention-staring, which is exactly
the methodological progression of the interpretability field.

## Reproduce

```bash
# NumPy (pedagogical anchor)
python examples/pointer_scaling.py            # sinusoidal PE, M=8..64
python examples/pointer_lpe_scaling.py        # learned PE, M=16/24/32
python examples/pointer_kitchen_sink.py       # M=16/24/32, 8000 iters
python examples/pointer_kitchen_sink_long.py  # M=32, 30000 iters
python examples/pointer_interp.py             # M=8 circuit analysis (post 7)

# PyTorch (scaling follow-up)
python examples/pytorch/pointer_sweep.py        # M=32, 4 configs, 15000 iters
python examples/pytorch/pointer_depth.py --layers 3 --iters 40000  # extended
python examples/pytorch/pointer_interp_deep.py  # M=32 3L mechanism (post 7)
```
