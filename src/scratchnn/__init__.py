"""scratchnn -- a pedagogical neural network library in pure Python.

Teaches logistic regression, softmax regression, and the multi-layer
perceptron as configurations of one Network class, built on hand-derived
per-layer backpropagation. The network computes logits; the loss interprets
them. Standard library only.

See docs/walkthrough.md for the accompanying narrative.
"""
from .neural_net import (
    dot,
    sigmoid,
    logsumexp,
    softmax,
    Layer,
    Linear,
    Tanh,
    ReLU,
    Loss,
    SigmoidBCE,
    SoftmaxCrossEntropy,
    Network,
    gradient_check,
)

__version__ = "0.1.0"

__all__ = [
    "dot",
    "sigmoid",
    "logsumexp",
    "softmax",
    "Layer",
    "Linear",
    "Tanh",
    "ReLU",
    "Loss",
    "SigmoidBCE",
    "SoftmaxCrossEntropy",
    "Network",
    "gradient_check",
]
