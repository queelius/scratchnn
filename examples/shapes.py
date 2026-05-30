"""Synthetic shape-classification demo for the CNN post.

Four shape templates (vertical bar, horizontal bar, two diagonals), each a
3x3 pattern with the same number of "on" pixels, placed at a random
top-left position on a 12x12 canvas. Position cardinality is identical
across classes, so pixel count carries no class signal: only the spatial
arrangement does.

The pedagogical trick is the *position-held-out split*: we generate one
sample per (class, position) pair across all 100 positions, then hold out
a random subset of positions for the test set. The training set never sees
any of the test positions.

  - An MLP has no parameter telling it "vertical bar at position (0, 0)
    and vertical bar at position (3, 7) are the same class." It memorizes
    training positions and fails on test positions.

  - A CNN with global average pooling is translation-invariant by
    construction: the conv learns a position-independent shape detector,
    and the pool discards position. It generalizes to held-out positions
    for free.

Run: python examples/shapes.py
"""
import random

import scratchnn as nn

CANVAS = 12
SHAPE = 3  # bounding-box side
N_POSITIONS = (CANVAS - SHAPE + 1) ** 2  # 10 * 10 = 100
N_CLASSES = 4

# Four 3x3 templates, each with 3 "on" pixels. Pixel count is constant
# across classes so the model cannot win on cardinality alone.
TEMPLATES = [
    # class 0: vertical bar in the middle column
    [[0, 1, 0],
     [0, 1, 0],
     [0, 1, 0]],
    # class 1: horizontal bar in the middle row
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],
    # class 2: diagonal \
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],
    # class 3: diagonal /
    [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]],
]


def render_image(flat):
    """Print a flat list[float] of length CANVAS*CANVAS as ASCII."""
    for r in range(CANVAS):
        line = "".join("#" if flat[r * CANVAS + c] > 0.5 else "."
                       for c in range(CANVAS))
        print("  " + line)


def make_image(cls, top_r, top_c):
    """Place TEMPLATES[cls] at (top_r, top_c) on a fresh CANVAS x CANVAS
    image of zeros. Returns flat list[float]."""
    canvas = [0.0] * (CANVAS * CANVAS)
    tmpl = TEMPLATES[cls]
    for i in range(SHAPE):
        for j in range(SHAPE):
            if tmpl[i][j]:
                canvas[(top_r + i) * CANVAS + (top_c + j)] = 1.0
    return canvas


def split_positions(seed=0, frac_train=0.25):
    """Return (train_positions, test_positions) as two disjoint lists of
    (top_r, top_c) tuples."""
    rng = random.Random(seed)
    positions = [(r, c)
                 for r in range(CANVAS - SHAPE + 1)
                 for c in range(CANVAS - SHAPE + 1)]
    rng.shuffle(positions)
    n_train = int(frac_train * len(positions))
    return positions[:n_train], positions[n_train:]


def make_dataset(positions):
    """Build (X, Y) over one image per (class, position) in positions."""
    X, Y = [], []
    for cls in range(N_CLASSES):
        for (r, c) in positions:
            X.append(make_image(cls, r, c))
            Y.append(cls)
    return X, Y


def accuracy(model, X, Y):
    hits = 0
    for x, y in zip(X, Y):
        probs = model.predict(x)
        if max(range(len(probs)), key=lambda i: probs[i]) == y:
            hits += 1
    return hits / len(X)


def param_count(model):
    return sum(len(v) for v, _ in model.parameters())


def train_and_report(label, model, X_tr, Y_tr, X_te, Y_te,
                     epochs, lr, batch_size):
    print(f"\n{'=' * 60}\n{label}  ({param_count(model)} params)\n{'=' * 60}")
    model.fit(X_tr, Y_tr, epochs=epochs, lr=lr, batch_size=batch_size,
              verbose=True)
    tr = accuracy(model, X_tr, Y_tr)
    te = accuracy(model, X_te, Y_te)
    print(f"  train acc: {tr:.4f}")
    print(f"  test  acc: {te:.4f}  (held-out positions)")
    return tr, te


def main(epochs=200, batch_size=16, seed=0,
         mlp_lr=0.1, cnn_lr=5.0):
    train_pos, test_pos = split_positions(seed=seed)
    X_tr, Y_tr = make_dataset(train_pos)
    X_te, Y_te = make_dataset(test_pos)
    print(f"shape dataset: 4 classes x {len(train_pos)} train positions "
          f"= {len(X_tr)} train samples")
    print(f"               4 classes x {len(test_pos)} test  positions "
          f"= {len(X_te)} test  samples")

    print("\nexample training image (class 0):")
    render_image(X_tr[0])

    # === MLP baseline ===
    random.seed(seed)
    mlp = nn.Network([nn.Linear(CANVAS * CANVAS, 32), nn.Tanh(),
                      nn.Linear(32, N_CLASSES)], nn.SoftmaxCrossEntropy())
    train_and_report("MLP (144-32-4)", mlp, X_tr, Y_tr, X_te, Y_te,
                     epochs, mlp_lr, batch_size)

    # === Translation-invariant CNN ===
    # The CNN needs a higher lr than the MLP because GlobalAvgPool divides
    # incoming gradients by H * W = 100, attenuating the signal that reaches
    # the conv kernel. lr = 5 compensates; lower lr stalls, lr = 10 diverges.
    random.seed(seed)
    conv = nn.Conv2D(in_channels=1, out_channels=4, kernel_size=SHAPE,
                     in_h=CANVAS, in_w=CANVAS)
    pool = nn.GlobalAvgPool(4, conv.out_h, conv.out_w)
    cnn = nn.Network([conv, nn.ReLU(), pool,
                      nn.Linear(pool.out_size, N_CLASSES)],
                     nn.SoftmaxCrossEntropy())
    train_and_report("CNN (Conv2D + ReLU + GlobalAvgPool + Linear)",
                     cnn, X_tr, Y_tr, X_te, Y_te,
                     epochs, cnn_lr, batch_size)


if __name__ == "__main__":
    main()
