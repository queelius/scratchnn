# scratchnn

A pedagogical neural network library in **pure Python** — no NumPy, no matrix
libraries, no autograd engine. It exists to be *read*.

`scratchnn` builds one idea up in three steps, all expressed as configurations
of a single `Network` class:

1. **Logistic regression** — one linear unit, a sigmoid, binary cross-entropy.
2. **Softmax regression** — the same idea across `K` classes.
3. **The multi-layer perceptron** — those units stacked into layers.

Gradients are hand-derived and computed by an explicit per-layer backward
pass; a numerical gradient checker verifies them. The companion
[`docs/series/01-foundations.md`](docs/series/01-foundations.md)
carries the math and the narrative. It is the first post in a series
on neural-network inductive biases; the rest of the series lives in
`docs/series/`.

## Install

```bash
pip install -e .              # the library -- standard library only
pip install -e ".[viz]"       # plus the optional matplotlib visualization
```

## Quick start

```python
import scratchnn as nn

# Logistic regression -- one linear layer
net = nn.Network([nn.Linear(2, 1)], loss=nn.SigmoidBCE())

# Softmax regression -- one linear layer, K outputs
net = nn.Network([nn.Linear(2, 3)], loss=nn.SoftmaxCrossEntropy())

# A multi-layer perceptron -- stack layers; same loss, same training
net = nn.Network([nn.Linear(2, 8), nn.Tanh(),
                  nn.Linear(8, 3)], loss=nn.SoftmaxCrossEntropy())

net.fit(X, Y, epochs=200, lr=0.1, batch_size=16)
net.predict(x)        # -> probability vector
```

## The organizing principle

The `Network` is the model: it maps an input to **logits** — raw, unnormalized
scores — and never computes a probability. The `Loss` interprets those logits:
it owns the output activation (sigmoid or softmax) and turns logits into a
scalar loss and a gradient. Hidden activations (`Tanh`, `ReLU`) are model
structure; the output activation is interpretation.

## Running things

```bash
python tests/test_neural_net.py    # 33 unit tests
python tests/test_gradients.py     # 4 gradient checks (analytic vs. finite differences)
python examples/demos.py           # logistic / softmax / XOR demos, ASCII decision boundaries
python -m scratchnn.visualize      # live (or headless-GIF) training visualization
```

The tests are plain `assert` scripts — no test framework required. Run them
after an editable install (`pip install -e .`).

## Layout

```
src/scratchnn/
    neural_net.py      the library
    visualize.py       optional matplotlib training visualization
tests/                 unit tests and gradient checks
examples/demos.py      runnable demos
docs/series/01-foundations.md       the pedagogical narrative (post 1)
docs/series/02-output-heads.md      output heads as inductive bias
docs/series/03-cnn.md               convolutional inductive bias
docs/series/04-fixed-context-lm.md  fixed-context language modeling
docs/series/05-rnn.md               recurrent inductive bias
docs/series/06-transformer-pointers.md   attention as content-addressable memory
docs/series/07-interpretability.md  reverse-engineering the pointer circuit
docs/series/08-rl-intro.md          intro to reinforcement learning
```

## License

Not yet chosen — add a `LICENSE` file and a `license` field to
`pyproject.toml`.
