"""Live training visualization for scratchnn.

Requires matplotlib, and Pillow for the headless GIF fallback. The core
library stays dependency-free; only this optional module needs them.

watch_training shows a live two-panel view -- decision boundary and loss
curve -- when an interactive matplotlib backend is available, and otherwise
writes an animated GIF.

Run the built-in XOR demo with: python -m scratchnn.visualize
"""
import matplotlib
import matplotlib.pyplot as plt

from . import neural_net as nn

_NONINTERACTIVE = {"agg", "pdf", "svg", "ps", "cairo", "template"}


def _is_headless():
    """True when matplotlib's active backend cannot open a window."""
    return matplotlib.get_backend().lower() in _NONINTERACTIVE


def _grid_value(net, gx, gy):
    """Predicted class index, or P(class 1) for a single-output network."""
    probs = net.predict([gx, gy])
    if len(probs) == 1:
        return probs[0]
    return max(range(len(probs)), key=lambda i: probs[i])


def _draw(ax_boundary, ax_loss, net, X, Y, history, x_range, y_range):
    """Redraw both panels for the current network state."""
    ax_boundary.clear()
    ax_loss.clear()

    cols = rows = 60
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    grid = []
    for r in range(rows):
        gy = y_lo + (y_hi - y_lo) * r / (rows - 1)
        grid.append([_grid_value(net,
                                 x_lo + (x_hi - x_lo) * c / (cols - 1), gy)
                     for c in range(cols)])
    ax_boundary.imshow(grid, origin="lower", extent=(x_lo, x_hi, y_lo, y_hi),
                       aspect="auto", alpha=0.6, cmap="coolwarm")
    ax_boundary.scatter([x[0] for x in X], [x[1] for x in X], c=Y,
                        cmap="coolwarm", edgecolors="black", s=30)
    ax_boundary.set_title(f"decision boundary (epoch {len(history)})")

    ax_loss.plot(range(len(history)), history)
    ax_loss.set_title("training loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("mean loss")


def _capture(fig):
    """Render the figure and return it as a PIL image."""
    from PIL import Image
    fig.canvas.draw()
    return Image.frombytes("RGBA", fig.canvas.get_width_height(),
                           bytes(fig.canvas.buffer_rgba()))


def watch_training(net, X, Y, epochs, lr, frames=50, gif_path="training.gif",
                   **fit_kwargs):
    """Train net on 2-D data X, Y while visualizing the progress.

    Shows a live matplotlib window if a display is available; otherwise
    writes an animated GIF to gif_path. Extra keyword arguments pass through
    to Network.fit. Returns the loss history.
    """
    xs = [x[0] for x in X]
    ys = [x[1] for x in X]
    pad_x = 0.25 * ((max(xs) - min(xs)) or 1.0)
    pad_y = 0.25 * ((max(ys) - min(ys)) or 1.0)
    x_range = (min(xs) - pad_x, max(xs) + pad_x)
    y_range = (min(ys) - pad_y, max(ys) + pad_y)

    headless = _is_headless()
    fig, (ax_boundary, ax_loss) = plt.subplots(1, 2, figsize=(11, 5))
    every = max(1, epochs // frames)
    snapshots = []

    if not headless:
        plt.ion()
        plt.show()

    def callback(epoch, network, history):
        if epoch % every != 0:
            return
        _draw(ax_boundary, ax_loss, network, X, Y, history, x_range, y_range)
        if headless:
            snapshots.append(_capture(fig))
        else:
            plt.pause(0.001)

    history = net.fit(X, Y, epochs, lr, callback=callback, **fit_kwargs)
    _draw(ax_boundary, ax_loss, net, X, Y, history, x_range, y_range)

    if headless:
        snapshots.append(_capture(fig))
        snapshots[0].save(gif_path, save_all=True,
                          append_images=snapshots[1:], duration=120, loop=0)
        plt.close(fig)
        print(f"headless: wrote {gif_path} ({len(snapshots)} frames)")
    else:
        plt.ioff()
        plt.show()
    return history


if __name__ == "__main__":
    import random
    random.seed(0)
    print("Training a 2-8-1 Tanh MLP on XOR with live visualization...")
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    labels = [0, 1, 1, 0]
    model = nn.Network([nn.Linear(2, 8), nn.Tanh(),
                        nn.Linear(8, 1)], nn.SigmoidBCE())
    watch_training(model, inputs, labels, epochs=4000, lr=0.5, batch_size=4)
