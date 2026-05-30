"""Convolutional layer for scratchnn.

A `Conv2D` layer slides a `kernel_size x kernel_size` kernel over a 2-D
input with shape `(in_channels, in_h, in_w)`, producing an output with
shape `(out_channels, out_h, out_w)` where
`out_h = in_h - kernel_size + 1` and similarly for `out_w`. Padding is
zero, stride is one. These restrictions keep the implementation small and
the math transparent; padding and strides are noted in the walkthrough but
not implemented here.

Inputs and outputs are presented as flat `list[float]` at the layer
boundary, the same way the rest of the library represents tensors. The 2-D
shape is implicit in the constructor arguments and reshaped via indexing,
not via nested lists. This keeps `Tanh`, `ReLU`, and `Linear` working
unchanged downstream.

Weight sharing across spatial positions is the entire point of the layer.
The same kernel weights apply at every position, so a single learned kernel
contributes to every output cell. Backward sums per-position gradients into
the shared kernel, which is the existing `+=` accumulation pattern the
library already uses for mini-batch gradients in `Linear`, just applied
over more axes.
"""
import math
import random

from .neural_net import Layer


class Conv2D(Layer):
    """A 2-D convolution with no padding and stride one.

    The input is flat `list[float]` of length `in_channels * in_h * in_w`,
    indexed as `c * in_h * in_w + r * in_w + col` for channel `c`, row `r`,
    column `col`. The output is flat `list[float]` of length
    `out_channels * out_h * out_w` indexed the same way.

    Parameters are stored as one flat weight vector per output kernel
    (length `in_channels * kernel_size * kernel_size`) plus a flat bias
    vector (length `out_channels`). `parameters()` yields one
    `(values, grads)` pair per kernel plus one for the bias, matching the
    "one pair per neuron" convention of `Linear`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, in_h, in_w):
        if in_h - kernel_size + 1 <= 0 or in_w - kernel_size + 1 <= 0:
            raise ValueError(
                f"kernel_size={kernel_size} too large for input "
                f"{in_h}x{in_w}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.in_h = in_h
        self.in_w = in_w
        self.out_h = in_h - kernel_size + 1
        self.out_w = in_w - kernel_size + 1
        self.out_size = out_channels * self.out_h * self.out_w

        # Xavier-style init scaled by fan-in.
        fan_in = in_channels * kernel_size * kernel_size
        r = 1.0 / math.sqrt(fan_in)
        self.kernels = [[random.uniform(-r, r) for _ in range(fan_in)]
                        for _ in range(out_channels)]
        self.bias = [0.0 for _ in range(out_channels)]
        self.dkernels = [[0.0 for _ in range(fan_in)]
                         for _ in range(out_channels)]
        self.dbias = [0.0 for _ in range(out_channels)]
        self.x = None  # cached flat input

    def forward(self, x):
        self.x = x
        k = self.k
        in_w = self.in_w
        in_hw = self.in_h * self.in_w
        out_w = self.out_w
        out_hw = self.out_h * self.out_w
        out = [0.0] * (self.out_channels * out_hw)
        for c_out in range(self.out_channels):
            kernel = self.kernels[c_out]
            b = self.bias[c_out]
            for r in range(self.out_h):
                for col in range(self.out_w):
                    z = b
                    for c_in in range(self.in_channels):
                        c_in_offset_x = c_in * in_hw
                        c_in_offset_k = c_in * k * k
                        for i in range(k):
                            row_offset_x = c_in_offset_x + (r + i) * in_w
                            row_offset_k = c_in_offset_k + i * k
                            for j in range(k):
                                z += (kernel[row_offset_k + j]
                                      * x[row_offset_x + col + j])
                    out[c_out * out_hw + r * out_w + col] = z
        return out

    def backward(self, g):
        k = self.k
        in_w = self.in_w
        in_hw = self.in_h * self.in_w
        out_w = self.out_w
        out_hw = self.out_h * self.out_w
        dx = [0.0] * len(self.x)
        for c_out in range(self.out_channels):
            kernel = self.kernels[c_out]
            dkernel = self.dkernels[c_out]
            for r in range(self.out_h):
                for col in range(self.out_w):
                    g_rc = g[c_out * out_hw + r * out_w + col]
                    self.dbias[c_out] += g_rc
                    for c_in in range(self.in_channels):
                        c_in_offset_x = c_in * in_hw
                        c_in_offset_k = c_in * k * k
                        for i in range(k):
                            row_offset_x = c_in_offset_x + (r + i) * in_w
                            row_offset_k = c_in_offset_k + i * k
                            for j in range(k):
                                xidx = row_offset_x + col + j
                                kidx = row_offset_k + j
                                dkernel[kidx] += g_rc * self.x[xidx]
                                dx[xidx] += g_rc * kernel[kidx]
        return dx

    def parameters(self):
        pairs = list(zip(self.kernels, self.dkernels))
        pairs.append((self.bias, self.dbias))
        return pairs


class GlobalAvgPool(Layer):
    """Global average pooling: collapse each channel's spatial extent to a
    single scalar by averaging.

    Input is flat `list[float]` of length `channels * height * width`
    (same row-major, channel-major layout as `Conv2D`). Output is flat
    `list[float]` of length `channels`, where `output[c]` is the mean of
    all `height * width` spatial cells of channel `c`.

    No parameters. Backward distributes the gradient on `output[c]`
    uniformly across the `height * width` input cells of channel `c`,
    scaled by `1 / (height * width)`.

    This layer is what turns a translation-equivariant convolutional stack
    into a translation-invariant classifier: the conv produces feature maps
    whose positions track the input, and the pool reads the average without
    caring where the activation was.
    """

    def __init__(self, channels, height, width):
        self.channels = channels
        self.h = height
        self.w = width
        self.hw = height * width
        self.in_size = channels * self.hw
        self.out_size = channels

    def forward(self, x):
        n = self.hw
        return [sum(x[c * n:(c + 1) * n]) / n for c in range(self.channels)]

    def backward(self, g):
        n = self.hw
        dx = [0.0] * self.in_size
        for c in range(self.channels):
            g_c = g[c] / n
            base = c * n
            for i in range(n):
                dx[base + i] = g_c
        return dx
