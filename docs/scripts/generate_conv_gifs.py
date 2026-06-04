#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Colour palette (neutral, consistent with existing docs)
# ---------------------------------------------------------------------------
COLOR_EMPTY = "#f0f0f0"  # inactive cell background
COLOR_GRID_LINE = "#cccccc"  # grid lines
COLOR_ACTIVE_IN = "#4a90d9"  # active input voxel
COLOR_ACTIVE_OUT = "#e07040"  # active output voxel
COLOR_KERNEL = "#ffd54f"  # kernel highlight (semi-transparent)
COLOR_KERNEL_EDGE = "#f9a825"  # kernel border
COLOR_TARGET = "#e07040"  # dashed outline for target output cells
COLOR_TEXT = "#333333"  # text / labels

GRID_SIZE = 8  # input grid is 8×8
STRIDE = 2
KERNEL_SIZE = 3
PAD = KERNEL_SIZE // 2  # border of inactive cells around each displayed grid

# A hand-picked sparse activation pattern on the 8×8 input grid
INPUT_ACTIVE = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=bool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_grid_top_left(grid, pad=PAD):
    """Pad a bool grid with *pad* rows on top and *pad* cols on the left."""
    return np.pad(grid, ((pad, 0), (pad, 0)), mode="constant", constant_values=False)


def _draw_grid(ax, grid_active, cell_size, origin, color_active, alpha=1.0, label=None, label_y_offset=0.0):
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
                (x, y),
                cell_size,
                cell_size,
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
        ax.text(cx, cy, label, ha="center", va="top", fontsize=11, fontweight="bold", color=COLOR_TEXT)


def _draw_target_outlines(ax, target_active, cell_size, origin):
    """Draw dashed outlines on cells that will become active in the output."""
    rows, cols = target_active.shape
    ox, oy = origin
    for r in range(rows):
        for c in range(cols):
            if not target_active[r, c]:
                continue
            x = ox + c * cell_size
            y = oy + (rows - 1 - r) * cell_size
            rect = patches.FancyBboxPatch(
                (x, y),
                cell_size,
                cell_size,
                boxstyle="round,pad=0.02",
                linewidth=1.5,
                edgecolor=COLOR_TARGET,
                facecolor="none",
                linestyle="--",
                zorder=2,
            )
            ax.add_patch(rect)


def _draw_legend(ax, x, y):
    """Draw a compact legend showing input, output, target outline, and inactive meanings."""
    s = 0.35  # swatch size
    gap = 0.15
    text_offset = s + 0.1

    items = [
        (COLOR_ACTIVE_IN, COLOR_GRID_LINE, "solid", "Input voxel"),
        (COLOR_ACTIVE_OUT, COLOR_GRID_LINE, "solid", "Output voxel"),
        (COLOR_EMPTY, "#999999", "solid", "Inactive voxel"),
        ("none", COLOR_TARGET, "dashed", "Target topology"),
    ]
    for i, (fcolor, ecolor, style, label) in enumerate(items):
        iy = y - i * (s + gap)
        rect = patches.FancyBboxPatch(
            (x, iy),
            s,
            s,
            boxstyle="round,pad=0.01",
            linewidth=1.5 if style == "dashed" else 1.0,
            edgecolor=ecolor,
            facecolor=fcolor,
            linestyle="--" if style == "dashed" else "-",
        )
        ax.add_patch(rect)
        ax.text(x + text_offset, iy + s / 2, label, va="center", fontsize=8.5, color=COLOR_TEXT)


def _draw_kernel_highlight(ax, centre_row, centre_col, grid_rows, grid_cols, cell_size, origin):
    """Draw a translucent rectangle showing the full kernel footprint.

    The highlight always covers KERNEL_SIZE cells and may extend beyond the
    grid boundary — this communicates that the kernel shape is fixed and
    out-of-bounds neighbours are simply absent (zero) in a sparse grid.
    """
    half_k = KERNEL_SIZE // 2
    r0 = centre_row - half_k
    r1 = centre_row + half_k + 1
    c0 = centre_col - half_k
    c1 = centre_col + half_k + 1
    ox, oy = origin
    x = ox + c0 * cell_size
    y = oy + (grid_rows - r1) * cell_size
    w = (c1 - c0) * cell_size
    h = (r1 - r0) * cell_size
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04",
        linewidth=2.5,
        edgecolor=COLOR_KERNEL_EDGE,
        facecolor=COLOR_KERNEL,
        alpha=0.45,
        zorder=5,
    )
    ax.add_patch(rect)
    # Darker centre cell to show which voxel the kernel is centred on
    cx = ox + centre_col * cell_size
    cy = oy + (grid_rows - 1 - centre_row) * cell_size
    centre = patches.FancyBboxPatch(
        (cx, cy),
        cell_size,
        cell_size,
        boxstyle="round,pad=0.02",
        linewidth=0,
        facecolor="#e6b800",
        alpha=0.55,
        zorder=6,
    )
    ax.add_patch(centre)
    return [rect, centre]


# ---------------------------------------------------------------------------
# Compute which output cells are active for strided conv
# ---------------------------------------------------------------------------


def _compute_strided_output(input_active, kernel_size, stride):
    """Return a bool array for the output grid of a strided sparse convolution.

    Scatters from active input voxels to find which output positions receive
    contributions — faithful to how sparse convolution works.  For each active
    input, every kernel offset that lands on a stride-aligned position produces
    an output voxel.
    """
    rows, cols = input_active.shape
    half_k = kernel_size // 2
    output_coords = set()
    for r in range(rows):
        for c in range(cols):
            if not input_active[r, c]:
                continue
            for dr in range(-half_k, half_k + 1):
                for dc in range(-half_k, half_k + 1):
                    dst_r = r + dr
                    dst_c = c + dc
                    if dst_r % stride == 0 and dst_c % stride == 0:
                        output_coords.add((dst_r // stride, dst_c // stride))
    if not output_coords:
        return np.zeros((0, 0), dtype=bool)
    min_r = min(o[0] for o in output_coords)
    max_r = max(o[0] for o in output_coords)
    min_c = min(o[1] for o in output_coords)
    max_c = max(o[1] for o in output_coords)
    out_rows = max_r - min_r + 1
    out_cols = max_c - min_c + 1
    output = np.zeros((out_rows, out_cols), dtype=bool)
    for r, c in output_coords:
        output[r - min_r, c - min_c] = True
    return output


def _compute_transposed_output(coarse_active, fine_active, kernel_size, stride):
    """Return a bool mask of target voxels that receive any contribution.

    In fVDB, the output topology of a strided transposed convolution is the
    full `target_grid` — voxels with no contributing source neighbors simply
    receive zero.  For the tutorial visualisation we compute which target
    voxels actually receive a contribution, so the animation can show them
    filling in as each coarse input cell is processed.
    """
    coarse_rows, coarse_cols = coarse_active.shape
    fine_rows, fine_cols = fine_active.shape
    half_k = kernel_size // 2
    output = np.zeros_like(fine_active, dtype=bool)
    for cr in range(coarse_rows):
        for cc in range(coarse_cols):
            if not coarse_active[cr, cc]:
                continue
            for kr in range(-half_k, half_k + 1):
                for kc in range(-half_k, half_k + 1):
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
    half_k = KERNEL_SIZE // 2
    positions = []
    for orow in range(out_rows):
        for ocol in range(out_cols):
            if coarse_output[orow, ocol]:
                positions.append((orow, ocol))

    # Padded input for display (extra row on top, extra col on left for kernel overhang)
    in_padded = _pad_grid_top_left(INPUT_ACTIVE)
    in_disp_rows, in_disp_cols = in_padded.shape

    cell_in = 0.9
    cell_out = 0.9 * STRIDE  # keep output cells visually larger
    gap = 2.5
    in_origin = (0, 0)
    out_origin = (
        in_disp_cols * cell_in + gap,
        (in_disp_rows * cell_in - out_rows * cell_out) / 2,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    total_w = out_origin[0] + out_cols * cell_out + 0.5
    total_h = in_disp_rows * cell_in + 0.5
    ax.set_xlim(-0.5, total_w)
    ax.set_ylim(-3.4, total_h)

    ax.set_title(
        "Strided Sparse Convolution  (kernel 3×3, stride 2)", fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12
    )

    # Draw an arrow between the two grids
    arrow_x = in_disp_cols * cell_in + gap / 2
    arrow_y = in_disp_rows * cell_in / 2
    ax.annotate(
        "",
        xy=(arrow_x + 0.6, arrow_y),
        xytext=(arrow_x - 0.6, arrow_y),
        arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2", color=COLOR_TEXT, lw=2),
    )

    # Pre-draw static grids (input padded, output not padded)
    _draw_grid(ax, in_padded, cell_in, in_origin, COLOR_ACTIVE_IN, label="Input (fine grid)", label_y_offset=0)
    empty_out = np.zeros_like(coarse_output)
    _draw_grid(ax, empty_out, cell_out, out_origin, COLOR_ACTIVE_OUT, label="Output (coarse grid)", label_y_offset=0)
    # Dashed outlines showing where output voxels will land
    _draw_target_outlines(ax, coarse_output, cell_out, out_origin)

    # Legend (below the output grid label)
    _draw_legend(ax, out_origin[0], out_origin[1] - 1.8)

    # Dynamic artists per frame
    dynamic_artists = []
    lead_in = 3  # frames showing only dashed outlines before animation starts

    def _init():
        return []

    def _animate(frame_idx):
        # Remove previous dynamic patches
        for a in dynamic_artists:
            a.remove()
        dynamic_artists.clear()

        step = frame_idx - lead_in  # negative during lead-in
        if 0 <= step < len(positions):
            orow, ocol = positions[step]
            # Kernel highlight on input — shift by PAD into padded display coords
            k = _draw_kernel_highlight(
                ax, orow * STRIDE + PAD, ocol * STRIDE + PAD, in_disp_rows, in_disp_cols, cell_in, in_origin
            )
            dynamic_artists.extend(k)

            # Corresponding highlight on the output cell (not padded)
            ox_h = out_origin[0] + ocol * cell_out
            oy_h = out_origin[1] + (out_rows - 1 - orow) * cell_out
            out_highlight = patches.FancyBboxPatch(
                (ox_h, oy_h),
                cell_out,
                cell_out,
                boxstyle="round,pad=0.04",
                linewidth=2.5,
                edgecolor=COLOR_KERNEL_EDGE,
                facecolor=COLOR_KERNEL,
                alpha=0.45,
                zorder=5,
            )
            ax.add_patch(out_highlight)
            dynamic_artists.append(out_highlight)

        # Redraw activated output cells up to current step
        for i in range(max(0, min(step + 1, len(positions)))):
            orow, ocol = positions[i]
            ox = out_origin[0] + ocol * cell_out
            oy = out_origin[1] + (out_rows - 1 - orow) * cell_out
            rect = patches.FancyBboxPatch(
                (ox, oy),
                cell_out,
                cell_out,
                boxstyle="round,pad=0.02",
                linewidth=0.5,
                edgecolor=COLOR_GRID_LINE,
                facecolor=COLOR_ACTIVE_OUT,
                zorder=3,
            )
            ax.add_patch(rect)
            dynamic_artists.append(rect)

        return dynamic_artists

    total_frames = lead_in + len(positions) + 6  # lead-in + animation + pause
    anim = animation.FuncAnimation(
        fig,
        _animate,
        init_func=_init,
        frames=total_frames,
        interval=500,
        blit=False,
    )
    anim.save(save_path, writer="pillow", fps=2, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")
    return coarse_output


def _build_transposed_conv_animation(save_path, coarse_output):
    """Build and save the strided transposed sparse convolution GIF."""
    fine_output = _compute_transposed_output(coarse_output, INPUT_ACTIVE, KERNEL_SIZE, STRIDE)
    coarse_rows, coarse_cols = coarse_output.shape

    # Enumerate kernel positions (coarse active cells)
    positions = []
    for cr in range(coarse_rows):
        for cc in range(coarse_cols):
            if coarse_output[cr, cc]:
                positions.append((cr, cc))

    # Padded fine grid for display (kernel overhang on top/left)
    fine_padded = _pad_grid_top_left(INPUT_ACTIVE)
    fine_disp_rows, fine_disp_cols = fine_padded.shape

    cell_coarse = 0.9 * STRIDE
    cell_fine = 0.9
    gap = 2.5
    coarse_origin = (0, (fine_disp_rows * cell_fine - coarse_rows * cell_coarse) / 2)
    fine_origin = (coarse_cols * cell_coarse + gap, 0)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    total_w = fine_origin[0] + fine_disp_cols * cell_fine + 0.5
    total_h = max(fine_disp_rows * cell_fine, coarse_origin[1] + coarse_rows * cell_coarse) + 0.5
    ax.set_xlim(-0.5, total_w)
    ax.set_ylim(-3.4, total_h)

    ax.set_title(
        "Strided Transposed Sparse Convolution  (kernel 3×3, stride 2)",
        fontsize=13,
        fontweight="bold",
        color=COLOR_TEXT,
        pad=12,
    )

    # Arrow
    arrow_x = coarse_cols * cell_coarse + gap / 2
    arrow_y = fine_disp_rows * cell_fine / 2
    ax.annotate(
        "",
        xy=(arrow_x + 0.6, arrow_y),
        xytext=(arrow_x - 0.6, arrow_y),
        arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2", color=COLOR_TEXT, lw=2),
    )

    # Static grids (coarse input not padded, fine output padded)
    _draw_grid(
        ax, coarse_output, cell_coarse, coarse_origin, COLOR_ACTIVE_IN, label="Input (coarse grid)", label_y_offset=0
    )
    empty_fine = np.zeros_like(fine_padded)
    _draw_grid(
        ax,
        empty_fine,
        cell_fine,
        fine_origin,
        COLOR_ACTIVE_OUT,
        label="Output (fine grid, guided by target_grid)",
        label_y_offset=0,
    )
    # Dashed outlines showing the full target_grid topology (padded display)
    _draw_target_outlines(ax, fine_padded, cell_fine, fine_origin)

    # Legend (below the coarse input grid label)
    _draw_legend(ax, coarse_origin[0], coarse_origin[1] - 1.8)

    # Cumulative set of activated fine cells
    activated = set()
    dynamic_artists = []
    lead_in = 3  # frames showing only dashed outlines before animation starts

    def _init():
        return []

    def _animate(frame_idx):
        for a in dynamic_artists:
            a.remove()
        dynamic_artists.clear()

        step = frame_idx - lead_in  # negative during lead-in
        if 0 <= step < len(positions):
            cr, cc = positions[step]
            # Highlight current coarse cell (not padded)
            cx = coarse_origin[0] + cc * cell_coarse
            cy = coarse_origin[1] + (coarse_rows - 1 - cr) * cell_coarse
            highlight = patches.FancyBboxPatch(
                (cx, cy),
                cell_coarse,
                cell_coarse,
                boxstyle="round,pad=0.04",
                linewidth=2.5,
                edgecolor=COLOR_KERNEL_EDGE,
                facecolor=COLOR_KERNEL,
                alpha=0.45,
                zorder=5,
            )
            ax.add_patch(highlight)
            dynamic_artists.append(highlight)

            # Highlight the corresponding kernel footprint on the fine output (padded display)
            out_artists = _draw_kernel_highlight(
                ax, cr * STRIDE + PAD, cc * STRIDE + PAD, fine_disp_rows, fine_disp_cols, cell_fine, fine_origin
            )
            for a in out_artists:
                a.set_alpha(0.35)
                a.set_zorder(4)
            dynamic_artists.extend(out_artists)

            # Mark fine cells produced by this coarse cell
            half_k = KERNEL_SIZE // 2
            for kr in range(-half_k, half_k + 1):
                for kc in range(-half_k, half_k + 1):
                    fr = cr * STRIDE + kr
                    fc = cc * STRIDE + kc
                    if 0 <= fr < GRID_SIZE and 0 <= fc < GRID_SIZE:
                        if fine_output[fr, fc]:
                            activated.add((fr, fc))

        # Draw all activated fine cells (shifted to padded display coords)
        for fr, fc in activated:
            fx = fine_origin[0] + (fc + PAD) * cell_fine
            fy = fine_origin[1] + (fine_disp_rows - 1 - (fr + PAD)) * cell_fine
            rect = patches.FancyBboxPatch(
                (fx, fy),
                cell_fine,
                cell_fine,
                boxstyle="round,pad=0.02",
                linewidth=0.5,
                edgecolor=COLOR_GRID_LINE,
                facecolor=COLOR_ACTIVE_OUT,
                zorder=3,
            )
            ax.add_patch(rect)
            dynamic_artists.append(rect)

        return dynamic_artists

    total_frames = lead_in + len(positions) + 6
    anim = animation.FuncAnimation(
        fig,
        _animate,
        init_func=_init,
        frames=total_frames,
        interval=500,
        blit=False,
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
