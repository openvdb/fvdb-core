#!/usr/bin/env python3
"""Generate animated GIF visualizations of strided sparse convolution
and strided transposed sparse convolution for the fVDB tutorials.

Usage:
    python docs/scripts/generate_conv_gifs.py

Outputs:
    docs/imgs/fig/strided_sparse_conv.gif
    docs/imgs/fig/strided_transposed_sparse_conv.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Colour palette (neutral, consistent with existing docs)
# ---------------------------------------------------------------------------
COLOR_EMPTY = "#f0f0f0"        # inactive cell background
COLOR_GRID_LINE = "#cccccc"    # grid lines
COLOR_ACTIVE_IN = "#4a90d9"    # active input voxel
COLOR_ACTIVE_OUT = "#e07040"   # active output voxel
COLOR_KERNEL = "#ffd54f"       # kernel highlight (semi-transparent)
COLOR_KERNEL_EDGE = "#f9a825"  # kernel border
COLOR_TEXT = "#333333"         # text / labels

GRID_SIZE = 8       # input grid is 8×8
STRIDE = 2
KERNEL_SIZE = 3

# A hand-picked sparse activation pattern on the 8×8 input grid
INPUT_ACTIVE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=bool)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_grid(ax, grid_active, cell_size, origin, color_active, alpha=1.0,
               label=None, label_y_offset=0.0):
    """Draw a 2-D sparse grid on *ax*.

    Parameters
    ----------
    grid_active : 2-D bool array (rows × cols), True = active voxel
    cell_size   : side length of one cell in data coordinates
    origin      : (x, y) of the bottom-left corner of the grid
    """
    rows, cols = grid_active.shape
    ox, oy = origin
    for r in range(rows):
        for c in range(cols):
            x = ox + c * cell_size
            # Flip row so row-0 is at the top
            y = oy + (rows - 1 - r) * cell_size
            color = color_active if grid_active[r, c] else COLOR_EMPTY
            rect = patches.FancyBboxPatch(
                (x, y), cell_size, cell_size,
                boxstyle="round,pad=0.02",
                linewidth=0.5,
                edgecolor=COLOR_GRID_LINE,
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)
    if label:
        cx = ox + cols * cell_size / 2
        cy = oy - 0.4 + label_y_offset
        ax.text(cx, cy, label, ha="center", va="top",
                fontsize=11, fontweight="bold", color=COLOR_TEXT)


def _draw_kernel_highlight(ax, row, col, grid_rows, cell_size, origin):
    """Draw a translucent rectangle showing the kernel position."""
    ox, oy = origin
    x = ox + col * cell_size
    y = oy + (grid_rows - 1 - (row + KERNEL_SIZE - 1)) * cell_size
    rect = patches.FancyBboxPatch(
        (x, y),
        KERNEL_SIZE * cell_size,
        KERNEL_SIZE * cell_size,
        boxstyle="round,pad=0.04",
        linewidth=2.5,
        edgecolor=COLOR_KERNEL_EDGE,
        facecolor=COLOR_KERNEL,
        alpha=0.45,
        zorder=5,
    )
    ax.add_patch(rect)
    return rect


# ---------------------------------------------------------------------------
# Compute which output cells are active for strided conv
# ---------------------------------------------------------------------------

def _compute_strided_output(input_active, kernel_size, stride):
    """Return a bool array for the output grid of a strided sparse convolution.

    An output cell at (or, oc) is active if *any* input cell covered by the
    kernel centred at the corresponding input position is active.
    """
    rows, cols = input_active.shape
    out_rows = (rows - kernel_size) // stride + 1
    out_cols = (cols - kernel_size) // stride + 1
    output = np.zeros((out_rows, out_cols), dtype=bool)
    for orow in range(out_rows):
        for ocol in range(out_cols):
            ir = orow * stride
            ic = ocol * stride
            patch = input_active[ir:ir + kernel_size, ic:ic + kernel_size]
            if patch.any():
                output[orow, ocol] = True
    return output


def _compute_transposed_output(coarse_active, fine_active, kernel_size, stride):
    """Return a bool array for the output of strided transposed sparse convolution.

    In fVDB, the output topology is specified by `out_grid`. For the tutorial
    visualisation we use the original fine grid as the target topology, so the
    output is the intersection of what the transposed conv *could* produce and
    the fine target grid.
    """
    coarse_rows, coarse_cols = coarse_active.shape
    fine_rows, fine_cols = fine_active.shape
    output = np.zeros_like(fine_active, dtype=bool)
    for cr in range(coarse_rows):
        for cc in range(coarse_cols):
            if not coarse_active[cr, cc]:
                continue
            for kr in range(kernel_size):
                for kc in range(kernel_size):
                    fr = cr * stride + kr
                    fc = cc * stride + kc
                    if 0 <= fr < fine_rows and 0 <= fc < fine_cols:
                        if fine_active[fr, fc]:
                            output[fr, fc] = True
    return output


# ---------------------------------------------------------------------------
# Animation builders
# ---------------------------------------------------------------------------

def _build_strided_conv_animation(save_path):
    """Build and save the strided sparse convolution GIF."""
    coarse_output = _compute_strided_output(INPUT_ACTIVE, KERNEL_SIZE, STRIDE)
    out_rows, out_cols = coarse_output.shape

    # Enumerate kernel positions that touch at least one active input cell
    positions = []
    for orow in range(out_rows):
        for ocol in range(out_cols):
            ir, ic = orow * STRIDE, ocol * STRIDE
            patch = INPUT_ACTIVE[ir:ir + KERNEL_SIZE, ic:ic + KERNEL_SIZE]
            if patch.any():
                positions.append((ir, ic, orow, ocol))

    cell_in = 0.9
    cell_out = 0.9 * STRIDE  # keep output cells visually larger
    gap = 2.5
    in_origin = (0, 0)
    out_origin = (GRID_SIZE * cell_in + gap,
                  (GRID_SIZE * cell_in - out_rows * cell_out) / 2)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    total_w = out_origin[0] + out_cols * cell_out + 0.5
    total_h = GRID_SIZE * cell_in + 1.0
    ax.set_xlim(-0.5, total_w)
    ax.set_ylim(-1.0, total_h)

    ax.set_title("Strided Sparse Convolution  (kernel 3×3, stride 2)",
                 fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)

    # Draw an arrow between the two grids
    arrow_x = GRID_SIZE * cell_in + gap / 2
    arrow_y = GRID_SIZE * cell_in / 2
    ax.annotate("", xy=(arrow_x + 0.6, arrow_y),
                xytext=(arrow_x - 0.6, arrow_y),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=COLOR_TEXT, lw=2))

    # Pre-draw static grids
    _draw_grid(ax, INPUT_ACTIVE, cell_in, in_origin, COLOR_ACTIVE_IN,
               label="Input (fine grid)", label_y_offset=0)
    # Output grid starts empty (drawn dimmed); we'll overlay active cells
    empty_out = np.zeros_like(coarse_output)
    _draw_grid(ax, empty_out, cell_out, out_origin, COLOR_ACTIVE_OUT,
               label="Output (coarse grid)", label_y_offset=0)

    # Dynamic artists per frame
    dynamic_artists = []

    def _init():
        return []

    def _animate(frame_idx):
        # Remove previous dynamic patches
        for a in dynamic_artists:
            a.remove()
        dynamic_artists.clear()

        if frame_idx < len(positions):
            ir, ic, orow, ocol = positions[frame_idx]
            # Kernel highlight on input
            k = _draw_kernel_highlight(ax, ir, ic, GRID_SIZE, cell_in, in_origin)
            dynamic_artists.append(k)

        # Redraw activated output cells up to current frame
        for i in range(min(frame_idx + 1, len(positions))):
            _, _, orow, ocol = positions[i]
            ox = out_origin[0] + ocol * cell_out
            oy = out_origin[1] + (out_rows - 1 - orow) * cell_out
            rect = patches.FancyBboxPatch(
                (ox, oy), cell_out, cell_out,
                boxstyle="round,pad=0.02",
                linewidth=0.5,
                edgecolor=COLOR_GRID_LINE,
                facecolor=COLOR_ACTIVE_OUT,
                zorder=3,
            )
            ax.add_patch(rect)
            dynamic_artists.append(rect)

        return dynamic_artists

    total_frames = len(positions) + 6  # pause on last frame
    anim = animation.FuncAnimation(
        fig, _animate, init_func=_init,
        frames=total_frames, interval=500, blit=False,
    )
    anim.save(save_path, writer="pillow", fps=2, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")
    return coarse_output


def _build_transposed_conv_animation(save_path, coarse_output):
    """Build and save the strided transposed sparse convolution GIF."""
    fine_output = _compute_transposed_output(
        coarse_output, INPUT_ACTIVE, KERNEL_SIZE, STRIDE
    )
    coarse_rows, coarse_cols = coarse_output.shape

    # Enumerate kernel positions (coarse active cells)
    positions = []
    for cr in range(coarse_rows):
        for cc in range(coarse_cols):
            if coarse_output[cr, cc]:
                positions.append((cr, cc))

    cell_coarse = 0.9 * STRIDE
    cell_fine = 0.9
    gap = 2.5
    coarse_origin = (0, (GRID_SIZE * cell_fine - coarse_rows * cell_coarse) / 2)
    fine_origin = (coarse_cols * cell_coarse + gap, 0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    total_w = fine_origin[0] + GRID_SIZE * cell_fine + 0.5
    total_h = GRID_SIZE * cell_fine + 1.0
    ax.set_xlim(-0.5, total_w)
    ax.set_ylim(-1.0, total_h)

    ax.set_title("Strided Transposed Sparse Convolution  (kernel 3×3, stride 2)",
                 fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)

    # Arrow
    arrow_x = coarse_cols * cell_coarse + gap / 2
    arrow_y = GRID_SIZE * cell_fine / 2
    ax.annotate("", xy=(arrow_x + 0.6, arrow_y),
                xytext=(arrow_x - 0.6, arrow_y),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=COLOR_TEXT, lw=2))

    # Static grids
    _draw_grid(ax, coarse_output, cell_coarse, coarse_origin, COLOR_ACTIVE_IN,
               label="Input (coarse grid)", label_y_offset=0)
    empty_fine = np.zeros_like(INPUT_ACTIVE)
    _draw_grid(ax, empty_fine, cell_fine, fine_origin, COLOR_ACTIVE_OUT,
               label="Output (fine grid, guided by out_grid)", label_y_offset=0)

    # Cumulative set of activated fine cells
    activated = set()
    dynamic_artists = []

    def _init():
        return []

    def _animate(frame_idx):
        for a in dynamic_artists:
            a.remove()
        dynamic_artists.clear()

        if frame_idx < len(positions):
            cr, cc = positions[frame_idx]
            # Highlight current coarse cell
            cx = coarse_origin[0] + cc * cell_coarse
            cy = coarse_origin[1] + (coarse_rows - 1 - cr) * cell_coarse
            highlight = patches.FancyBboxPatch(
                (cx, cy), cell_coarse, cell_coarse,
                boxstyle="round,pad=0.04",
                linewidth=2.5,
                edgecolor=COLOR_KERNEL_EDGE,
                facecolor=COLOR_KERNEL,
                alpha=0.45,
                zorder=5,
            )
            ax.add_patch(highlight)
            dynamic_artists.append(highlight)

            # Mark fine cells produced by this coarse cell
            for kr in range(KERNEL_SIZE):
                for kc in range(KERNEL_SIZE):
                    fr = cr * STRIDE + kr
                    fc = cc * STRIDE + kc
                    if 0 <= fr < GRID_SIZE and 0 <= fc < GRID_SIZE:
                        if fine_output[fr, fc]:
                            activated.add((fr, fc))

        # Draw all activated fine cells
        for (fr, fc) in activated:
            fx = fine_origin[0] + fc * cell_fine
            fy = fine_origin[1] + (GRID_SIZE - 1 - fr) * cell_fine
            rect = patches.FancyBboxPatch(
                (fx, fy), cell_fine, cell_fine,
                boxstyle="round,pad=0.02",
                linewidth=0.5,
                edgecolor=COLOR_GRID_LINE,
                facecolor=COLOR_ACTIVE_OUT,
                zorder=3,
            )
            ax.add_patch(rect)
            dynamic_artists.append(rect)

        return dynamic_artists

    total_frames = len(positions) + 6
    anim = animation.FuncAnimation(
        fig, _animate, init_func=_init,
        frames=total_frames, interval=500, blit=False,
    )
    anim.save(save_path, writer="pillow", fps=2, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "imgs", "fig")
    os.makedirs(output_dir, exist_ok=True)

    strided_path = os.path.join(output_dir, "strided_sparse_conv.gif")
    transposed_path = os.path.join(output_dir, "strided_transposed_sparse_conv.gif")

    coarse_output = _build_strided_conv_animation(strided_path)
    _build_transposed_conv_animation(transposed_path, coarse_output)


if __name__ == "__main__":
    main()
