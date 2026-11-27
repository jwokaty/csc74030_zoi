"""Plotting helpers for pyramids, histograms, and model results."""
import matplotlib.pyplot as plt

# Plotting written by Claude Sonnet 5.5
# The histogram and entropy lines are mine.
def show_pyramid(p, title, w=None, h=None, hist=False, cols=4):
    rows = math.ceil(len(p) / cols)
    fig = plt.figure(figsize=(w, h) if w is not None and h is not None else None)
    # With histograms, each row of images is followed by a row of histograms
    total_rows = rows * 2 if hist else rows
    # Make every grid cell square so the images touch, and center the grid
    fig_w, fig_h = fig.get_size_inches()
    side = min(0.98 * fig_w / cols, 0.92 * fig_h / total_rows)
    grid_w, grid_h = side * cols / fig_w, side * total_rows / fig_h
    left, top = (1 - grid_w) / 2, 0.93
    grid = fig.add_gridspec(total_rows, cols, wspace=0, hspace=0,
                            left=left, right=left + grid_w, top=top, bottom=top - grid_h)
    for i, level in enumerate(p):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(grid[row * 2 if hist else row, col])
        ax.imshow(level, cmap="gray")
        ax.axis("off")
        if hist:
            cell = fig.add_subplot(grid[row * 2 + 1, col])
            cell.axis("off")
            ax_hist = cell.inset_axes([0.15, 0.2, 0.8, 0.6])
            counts, bin_edges = np.histogram(level.ravel(), bins=511, range=(-255, 256))
            ps = counts / counts.sum()
            # remove nonzero
            ps = ps[ps > 0]
            entropy = -sum(ps*np.log(ps))
            ax_hist.bar(bin_edges[:-1], counts, edgecolor="C0")
            ax_hist.tick_params(axis="both", labelsize=7)
            ax_hist.set_title(f"Level {i}, H(X) = {entropy:.2f}", fontsize=9)
    fig.suptitle(title, y=0.98)
    plt.show()
