# CNN Outline: Architecture as Inductive Bias (Post 2 of 4)

## Title and thesis

**Title:** Convolutional Networks: Locality and Translation Equivariance as Architecture

**One-line thesis:** A CNN is an MLP with two priors wired into its weights, locality (a unit looks at a small neighborhood) and translation equivariance (the same look is applied at every position); architecture is inductive bias made structural.

## Pedagogical arc

The reader should leave with these, in order:

1. The MLP from `scratchnn` already classifies 8x8 digits at roughly 98%. So why bother? Because it has no idea that pixel $(3, 4)$ and pixel $(3, 5)$ are neighbors. It treats the 64 inputs as 64 unrelated scalars. That is a fact about the *prior*, not about the *data*.
2. A convolution is a single linear unit (a *kernel*) reused at every position. The reuse is the entire idea. Everything else, channels, strides, padding, falls out.
3. The same weight applied at many positions means many fewer parameters and *automatic* translation equivariance: shift the input, shift the output.
4. Backpropagation through Conv2D is not a new derivation. It is the `Linear` backward seen earlier, *summed over positions*. The `+=` pattern that already accumulates over a mini-batch now accumulates over spatial positions too. Same operator, finer grain.
5. **Punchline:** A small CNN beats the MLP on the same dataset with roughly an order of magnitude fewer parameters, because its architecture already knows what the MLP had to learn from scratch. That is what an inductive bias *buys*.

## Section-by-section breakdown

### Section 1: The MLP's blind spot

- Recall: the MLP from §4 of the walkthrough hits ~98% on 8x8 UCI digits. Treat that as the baseline.
- Permute the 64 input pixels with a fixed random permutation, retrain. The MLP hits the same accuracy. The model is permutation-blind: it does not know which pixels are adjacent.
- For a human, the permuted digits are unreadable. For the MLP, nothing changed.
- That equivalence is a diagnostic. It says: the MLP's prior is "any input coordinate may interact with any other on equal footing." For images, that is too weak.
- The CNN is what you build when you commit to a stronger prior.

### Section 2: One unit, reused everywhere

- A `Linear` unit is one weight vector $\mathbf{w}$ and a bias $b$, scoring $z = \mathbf{w} \cdot \mathbf{x} + b$.
- A convolutional unit is the same thing, applied at every position of the input. Restrict $\mathbf{w}$ to a small **kernel** of size $k \times k$, and slide it across the image:

$$z_{r, c} = \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} W_{i, j}\, x_{r+i,\, c+j} + b.$$

- $W$ is shared across all $(r, c)$. There is one $W$, not one per position.
- That is the entire definition. The rest is bookkeeping: how to handle borders (padding), how far to slide between applications (stride), and how to stack multiple kernels in parallel (output channels).
- Output channels: $C_{\text{out}}$ independent kernels run on the same input, producing $C_{\text{out}}$ stacked feature maps. Input channels: a kernel has shape $(C_{\text{in}}, k, k)$ and dots through all input channels at each spatial position.

### Section 3: Weight sharing is the translation prior

- Translation equivariance, stated cleanly: if $x'$ is $x$ shifted by $(\Delta r, \Delta c)$, then the output of the convolutional layer on $x'$ is the output on $x$ shifted by the same $(\Delta r, \Delta c)$ (modulo borders).
- This is not a property the network *learns*. It is a property of the *operator*, true at initialization, true after training, true for every kernel.
- Why? Because the same $W$ is applied at every position. Shift the input by one, and every windowed dot product is computed at a position that is also shifted by one. The output simply slides.
- The MLP, by contrast, has no reason whatsoever to be equivariant. Each output unit has its own $\mathbf{w}$. Translating the input scrambles which weights see which pixels.
- This is the inductive bias, encoded structurally: weight sharing *is* translation equivariance.

### Section 4: The parameter count tells the story

- 8x8 digit input, 10 classes.
- MLP from walkthrough: roughly `64 -> 32 -> 10`. Parameters: $64 \cdot 32 + 32 + 32 \cdot 10 + 10 = 2 410$. Use as the baseline.
- CNN: one conv layer with 4 kernels of size $3 \times 3$ on 1 input channel, followed by flatten and a `Linear` head.
  - Conv parameters: $4 \cdot (3 \cdot 3 \cdot 1) + 4 = 40$.
  - With `padding=0` and `stride=1`, the conv output is $4 \times 6 \times 6 = 144$ units.
  - Head: `Linear(144, 10)`. Parameters: $144 \cdot 10 + 10 = 1 450$.
  - Total: $40 + 1 450 = 1 490$.
- That is the picture. Most of the model's capacity sits in 40 parameters of conv kernel, which see every position; the head is a thin reader on top. A more aggressive design (smaller head, average-pooled features) cuts the total to a few hundred.
- The point is structural: the conv layer is *cheap* because weight sharing collapses what would have been thousands of MLP parameters into a single reused kernel.

### Section 5: Forward pass, in code

- A `Conv2D` layer holds `kernels` of shape $(C_{\text{out}}, C_{\text{in}}, k, k)$ and `bias` of shape $(C_{\text{out}},)$. Each kernel is one weight tensor; together they generalize the list-of-weight-vectors layout of `Linear`.
- `forward(x)`, with input shaped $(C_{\text{in}}, H, W)$, returns output shaped $(C_{\text{out}}, H', W')$:

```
for c_out in range(C_out):
    for r in range(H'):
        for c in range(W'):
            z = bias[c_out]
            for c_in in range(C_in):
                for i in range(k):
                    for j in range(k):
                        z += kernels[c_out][c_in][i][j] * x[c_in][r+i][c+j]
            out[c_out][r][c] = z
```

- Five nested loops, no NumPy. That is the price of staying pedagogical. With $k = 3$, $H = W = 8$, $C_{\text{in}} = 1$, $C_{\text{out}} = 4$, this is roughly 1 300 multiply-adds per forward pass per example. Tractable.
- Cache `x` for the backward pass. Same pattern as `Linear`.

### Section 6: Backward pass, by careful chain rule

- Let $g_{c_{\text{out}}, r, c} = \partial L / \partial z_{c_{\text{out}}, r, c}$ be the incoming gradient on the output feature map. We need three things:

(a) Gradient with respect to the kernel weights $W_{c_{\text{out}}, c_{\text{in}}, i, j}$. Each weight participates in the output at *every* spatial position $(r, c)$. So the chain rule sums over positions:

$$\frac{\partial L}{\partial W_{c_{\text{out}}, c_{\text{in}}, i, j}}
  = \sum_{r, c} g_{c_{\text{out}}, r, c}\, x_{c_{\text{in}},\, r+i,\, c+j}.$$

  This is the `Linear` formula $g_i \, x_j$, summed over the spatial axes. The reuse of $W$ in forward is exactly what causes the gradient to accumulate in backward.

(b) Gradient with respect to the bias:

$$\frac{\partial L}{\partial b_{c_{\text{out}}}} = \sum_{r, c} g_{c_{\text{out}}, r, c}.$$

(c) Gradient with respect to the input pixel $x_{c_{\text{in}}, r', c'}$. The pixel was read by output positions $(r, c)$ with $r + i = r'$ and $c + j = c'$, that is $(r, c) = (r' - i, c' - j)$. Summing over all kernels and the positions where the pixel was used:

$$\frac{\partial L}{\partial x_{c_{\text{in}}, r', c'}}
  = \sum_{c_{\text{out}}} \sum_{i, j} W_{c_{\text{out}}, c_{\text{in}}, i, j}\, g_{c_{\text{out}},\, r' - i,\, c' - j},$$

  where terms with $(r' - i, c' - j)$ out of range are dropped. This is the spatial transpose of the forward, sometimes called a transposed convolution.

- The crucial observation: every accumulation here is the *same* `+=` pattern the library already uses for mini-batch gradients in `Linear`. The mini-batch sums per-example gradients; the conv sums per-position gradients. Same operator, finer grain. The walkthrough's closing note (per-layer to per-operation) is the same idea pulled in a different direction.

### Section 7: Pooling, as a smaller commentary

- Add an average-pool 2x2 with stride 2 after the conv layer. This trades a small drop in spatial resolution for a further reduction in head parameters.
- Pool has no weights. Its backward distributes the incoming gradient uniformly to the four input cells of each pool window.
- Optional in the post. Mentioned because it shows that *not all layers carry parameters*, the same lesson `Tanh` and `ReLU` already taught.

### Section 8: Training and results on UCI 8x8 digits

- Same training recipe as the MLP: mini-batch SGD, learning rate similar, same number of epochs, same train/test split.
- Report:
  - MLP baseline: ~98% test accuracy, ~2 410 parameters.
  - CNN: target ~98.5 to 99% test accuracy, ~1 490 parameters (or ~300 with average-pool head).
  - Permuted-input control: MLP unchanged (~98%); CNN collapses to roughly the MLP's permuted accuracy on small features, because the locality prior is now actively *wrong*.
- The permuted-input control is the load-bearing experiment. It is what makes the inductive-bias claim falsifiable. If the CNN matched its unpermuted accuracy on permuted pixels, weight sharing would have bought us nothing.

### Section 9: Inductive bias, named

- The CNN commits to two priors. Make them explicit:
  - **Locality:** A unit reads a $k \times k$ window, not the whole image. Long-range interactions only emerge by stacking.
  - **Translation equivariance:** The same weights at every position. Shift the input, shift the output.
- What is given up:
  - Free interaction between distant pixels. An MLP can learn "the upper-left corner depends on the lower-right corner" with one weight. A single conv layer cannot represent that at all; deeper conv stacks can, but indirectly.
  - Position-specific filters. If digit centering matters (it sometimes does), the CNN has to learn to compensate elsewhere or be helped with extra channels.
- This is the trade. An MLP is the most generic model: every input may interact with every other on equal footing. A CNN says: most useful interactions on images are local and translation-equivariant. That belief, *baked into the weights*, is what allows the CNN to learn from less data with fewer parameters.

## Worked example: 8x8 UCI digits, in concrete numbers

- Dataset: `sklearn.datasets.load_digits()`, 1 797 images of $8 \times 8$ grayscale digits, 10 classes. Same handoff as walkthrough §5.
- Architecture: `Conv2D(C_in=1, C_out=4, k=3, padding=0, stride=1) -> ReLU -> Flatten -> Linear(144, 10) -> SoftmaxCrossEntropy`.
- Parameters: 1 490.
- Hyperparameters: learning rate 0.05, batch size 16, 100 epochs.
- Targets:
  - Test accuracy: $\geq 98.5\%$, ideally $99\%$.
  - Permuted-input ablation: drops to $\leq 95\%$ (numbers approximate; the point is the gap, not the digit).
- Optional smaller variant for the "really fewer parameters" headline: `Conv2D(C_out=4, k=3) -> ReLU -> AvgPool(2x2) -> Flatten -> Linear(36, 10) -> Softmax`, ~410 parameters, target ~97%.

## Code approach: stay in pure Python, or switch?

**Recommendation: stay in pure Python.** Reasoning:

- The series' charter is pedagogical clarity. The whole reader expectation set by post 1 is "you can read the entire library." Introducing NumPy in post 2 burns that.
- A pure-Python `Conv2D` on 8x8 inputs is genuinely small: roughly 50 lines including forward, backward, and parameters. The five-nested-loop forward is unflattering but transparent; that transparency is *the* feature.
- Performance is adequate for this dataset. 1 797 examples, ~1 300 multiply-adds per conv forward, ~5x for backward, hundreds of epochs. Order of minutes on a laptop. Acceptable.
- This forces an honest accounting of when pure Python *would* stop working. The post should end with a short paragraph: "We stayed in pure Python for one reason: nothing in this layer demanded a tensor library. The forward is five nested loops; the backward is also five nested loops, with the indices rearranged. For $28 \times 28$ MNIST and three conv layers, those loops become untenable, and a vectorized implementation (NumPy, then PyTorch) is genuine relief, not premature optimization." Set up Post 3.

**Implementation sketch:**

- Add `Conv2D(Layer)` and a `Flatten(Layer)` to `neural_net.py`, or to a new `conv.py` if the core file's character must be preserved. (Author choice; either is defensible.)
- Reuse the existing `Layer` protocol, `forward(x) / backward(g) / parameters()`. The pair-of-flat-lists invariant still holds: kernels are stored as nested lists, but `parameters()` yields them as flat `list[float]` pairs the way `Linear` already does for `weights[i]`.
- Add `gradient_check` cases for `Conv2D` and `Flatten`. The kink caveat does not apply (Conv2D is everywhere differentiable); use a small `Tanh` head to avoid the ReLU-kink case if needed.

## Math content checklist

- Define 2D convolution with the explicit double sum.
- State and prove (in one paragraph) translation equivariance from weight sharing.
- Derive parameter count for `Conv2D(C_in, C_out, k)`: $C_{\text{out}} \cdot (C_{\text{in}} \cdot k^2) + C_{\text{out}}$.
- Compare against the equivalent `Linear` layer that would map the same flattened input to the same flattened output: $C_{\text{in}} H W \cdot C_{\text{out}} H' W' + C_{\text{out}} H' W'$. For $C_{\text{in}} = 1$, $H = W = 8$, $k = 3$, $C_{\text{out}} = 4$, that is $64 \cdot 144 + 144 = 9 360$, versus the conv's 40. A 234x reduction.
- Backward derivation for $W$, $b$, and $x$, with index ranges spelled out so the reader can write the five-loop code from the math.
- Connect to `scratchnn`'s existing `+=` accumulation pattern: conv backward extends the same accumulation, just over more axes.

## Figures to produce

Five figures, ranked by load-bearing:

1. **Kernel weight visualization.** The 4 learned 3x3 kernels rendered as small heatmaps. Earns its place because it is the most concrete answer to "what did the network learn?" Often the kernels look like oriented edge detectors. If they do not, that is interesting too.
2. **Feature maps for one example digit.** The input digit on the left, the 4 post-conv feature maps next to it. Shows weight sharing visually: the same kernel response, computed at every position.
3. **Accuracy vs parameter count (MLP, CNN, CNN-small).** Three points on a single plot. Same data, different inductive biases. This is the headline image for the inductive-bias claim.
4. **Permuted-input control.** Bar chart: MLP unpermuted, MLP permuted, CNN unpermuted, CNN permuted. The CNN's collapse on permuted input is the falsification test for the locality prior.
5. **Optional: receptive-field cartoon.** A small diagram showing one output cell highlighted, with the $k \times k$ input window it sees, then a deeper layer's output cell with its enlarged receptive field. Useful to motivate why depth in a CNN buys non-local interactions back, indirectly.

If only one figure makes the cut: Figure 3. If two: 1 and 3.

## Handoff to RNN post

The CNN encodes a prior about *space*: pixels nearby in 2D coordinates interact more than pixels far apart, and the same interaction rule applies at every position. Swap "space" for "time" and the same logic applies to sequences. Tokens nearby in time interact more than tokens far apart. The same parameters should apply at every step, because there is no privileged moment.

That is the recurrent network: a single transition shared across all time steps, just as a convolutional kernel is shared across all positions. Locality becomes "depends on the recent past." Translation equivariance becomes "time-shift equivariance," a model run on a delayed input produces a delayed output. The architecture is a different commitment, to a different symmetry, but it is the same move: bake the prior into the weights by reusing them.

Post 3 makes that move concrete on a sequence task, and inherits the same `+=` accumulation pattern the CNN extends: gradients now sum over time steps rather than spatial positions.

## Open questions for the author

- Conv2D in `neural_net.py`, or a new module? The core file is currently ~300 lines of unbroken pedagogical flow. Adding Conv2D doubles its size and changes its character. A separate `conv.py` (or `scratchnn.conv`) would keep the original walkthrough intact while still respecting the `Layer` protocol. Recommended, but worth a deliberate call.
- Should the post include the average-pool variant, or is the 1 490-parameter version cleaner? The smaller variant strengthens the parameter-efficiency claim but adds a second new layer type. Vote: keep it as an aside, not the headline.
- Permuted-input ablation: run-time on pure Python is the gating factor. If 100 epochs takes 5 minutes, fine. If 50, reconsider.
- Should the post include a brief mention of stride and padding beyond the worked example? Stride > 1 and padding are useful for the receptive-field discussion but expand the math. Suggest: define them in a short aside, do not use them in the worked example.
- Cross-link strategy: should the post link forward to the RNN and Transformer posts, or stand alone? The walkthrough did not forward-link to this post; consistency suggests no.
