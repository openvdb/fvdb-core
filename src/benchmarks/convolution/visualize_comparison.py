# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Visualization for the sparse convolution comparison benchmark.
#
# Reads JSON output from benchmark_sparse_conv_comparison.py and produces
# publication-quality plots comparing fVDB against spconv, torchsparse,
# MinkowskiEngine, and dense PyTorch conv3d.
#
# Usage:
#   python visualize_comparison.py --file comparison_results.json
#   python visualize_comparison.py --file results.json --filter "fVDB spconv"
#

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# =============================================================================
# 1. Loading
# =============================================================================

# Consistent color and marker assignments for each library so they are
# instantly recognizable across all plots.
LIBRARY_STYLE = {
    "fVDB": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "spconv": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "torchsparse": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "MinkowskiEngine": {"color": "#9467bd", "marker": "D", "linestyle": ":"},
    "Dense (conv3d)": {"color": "#d62728", "marker": "x", "linestyle": ":"},
}


def _style_for(library: str) -> dict:
    return LIBRARY_STYLE.get(library, {"color": "gray", "marker": ".", "linestyle": "-"})


def load_results(path: str) -> pd.DataFrame:
    """Load comparison benchmark JSON into a DataFrame."""
    with open(path, "r") as f:
        data = json.load(f)

    records = []
    for b in data.get("benchmarks", []):
        rec = {
            "library": b["library"],
            "suite": b["suite"],
            "num_voxels": b["num_voxels"],
            "setup_ms": b["setup_ms"],
            "mean_ms": b["mean_ms"],
            "std_ms": b["std_ms"],
            "min_ms": b["min_ms"],
            "max_ms": b["max_ms"],
        }
        rec.update(b.get("params", {}))
        records.append(rec)

    return pd.DataFrame(records)


# =============================================================================
# 2. Axis formatters
# =============================================================================


def format_voxels(x, _pos):
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    elif x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def format_ms(x, _pos):
    if x >= 1000:
        return f"{x / 1000:.1f}s"
    elif x >= 1:
        return f"{x:.1f}ms"
    elif x >= 0.001:
        return f"{x * 1000:.0f}us"
    return f"{x:.4f}ms"


# =============================================================================
# 3. Plot functions
# =============================================================================


def plot_grid_size_scaling(df: pd.DataFrame) -> None:
    """Time vs grid size, one line per library.  Log-log axes."""
    suite_df = df[df["suite"] == "grid_size"].copy()
    if suite_df.empty:
        print("No grid_size data to plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    for lib in suite_df["library"].unique():
        sub = suite_df[suite_df["library"] == lib].sort_values("num_voxels")
        s = _style_for(lib)
        ax.errorbar(
            sub["num_voxels"],
            sub["mean_ms"],
            yerr=sub["std_ms"],
            marker=s["marker"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=2.5,
            markersize=8,
            capsize=3,
            label=lib,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(format_voxels))
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.set_xlabel("Grid Size (voxels)")
    ax.set_ylabel("Forward Time")
    ax.set_title("Grid-Size Scaling (C=32, K=3x3x3)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out = "comparison_grid_size.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


def plot_sparsity_breakeven(df: pd.DataFrame) -> None:
    """Time vs occupancy at each bbox size -- the key comparison plot.

    Sparse libraries appear as lines; dense conv3d appears as a horizontal
    reference showing the cost of processing the full bounding box.
    """
    suite_df = df[df["suite"] == "sparsity"].copy()
    if suite_df.empty:
        print("No sparsity data to plot.")
        return

    bbox_sizes = sorted(suite_df["bbox_dim"].unique())
    sns.set_theme(style="whitegrid", context="talk")
    n_panels = len(bbox_sizes)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), squeeze=False)

    for col, bbox_dim in enumerate(bbox_sizes):
        ax = axes[0][col]
        bbox_df = suite_df[suite_df["bbox_dim"] == bbox_dim]

        # Dense reference line (100% occupancy from the Dense adapter, or
        # the dense adapter at the first available occupancy since it always
        # processes the full grid).
        dense_df = bbox_df[bbox_df["library"] == "Dense (conv3d)"]
        if not dense_df.empty:
            dense_time = dense_df["mean_ms"].iloc[0]
            ax.axhline(
                y=dense_time,
                color=_style_for("Dense (conv3d)")["color"],
                linestyle="--",
                linewidth=2,
                label=f"Dense conv3d ({dense_time:.2f} ms)",
            )

        # Sparse libraries
        sparse_libs = [lib for lib in bbox_df["library"].unique() if lib != "Dense (conv3d)"]
        for lib in sparse_libs:
            sub = bbox_df[bbox_df["library"] == lib].sort_values("occupancy_pct")
            s = _style_for(lib)
            ax.errorbar(
                sub["occupancy_pct"],
                sub["mean_ms"],
                yerr=sub["std_ms"],
                marker=s["marker"],
                color=s["color"],
                linestyle=s["linestyle"],
                linewidth=2.5,
                markersize=8,
                capsize=3,
                label=lib,
            )

        total = int(bbox_dim) ** 3
        ax.set_xlabel("Occupancy (%)")
        ax.set_ylabel("Forward Time")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
        ax.set_title(f"bbox={bbox_dim}^3 ({total:,} cells)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Sparsity Breakeven (C=32, K=3x3x3)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "comparison_sparsity_breakeven.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


def plot_channel_scaling(df: pd.DataFrame) -> None:
    """Time vs channel count, one line per library.  Log-log axes."""
    suite_df = df[df["suite"] == "channels"].copy()
    if suite_df.empty:
        print("No channel data to plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    for lib in suite_df["library"].unique():
        sub = suite_df[suite_df["library"] == lib].sort_values("channels")
        s = _style_for(lib)
        ax.errorbar(
            sub["channels"],
            sub["mean_ms"],
            yerr=sub["std_ms"],
            marker=s["marker"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=2.5,
            markersize=8,
            capsize=3,
            label=lib,
        )

    # O(C) and O(C^2) reference slopes
    chan_vals = np.sort(suite_df["channels"].unique())
    if len(chan_vals) >= 2:
        ref_t = suite_df[suite_df["channels"] == chan_vals[0]]["mean_ms"].mean()
        y_lin = ref_t * (chan_vals / chan_vals[0])
        y_quad = ref_t * (chan_vals / chan_vals[0]) ** 2
        ax.plot(chan_vals, y_lin, ":", alpha=0.4, color="gray", label="O(C) ref")
        ax.plot(chan_vals, y_quad, "-.", alpha=0.4, color="gray", label="O(C^2) ref")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Channels (C)")
    ax.set_ylabel("Forward Time")
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.set_title("Channel Scaling (16^3 grid, K=3x3x3)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out = "comparison_channel_scaling.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


def plot_topology_overhead(df: pd.DataFrame) -> None:
    """Stacked bar chart: setup cost vs forward cost per library.

    Uses the largest grid size from the grid_size suite to make the
    comparison visually clear.
    """
    suite_df = df[df["suite"] == "grid_size"].copy()
    if suite_df.empty:
        print("No grid_size data for topology overhead plot.")
        return

    max_dim = suite_df["grid_dim"].max()
    subset = suite_df[suite_df["grid_dim"] == max_dim].copy()
    if subset.empty:
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 5))

    libs = list(subset["library"])
    setup = list(subset["setup_ms"])
    forward = list(subset["mean_ms"])

    x = np.arange(len(libs))
    bar_width = 0.5

    bars_setup = ax.bar(x, setup, bar_width, label="Setup (topology)", color="#7fcdbb")
    bars_fwd = ax.bar(x, forward, bar_width, bottom=setup, label="Forward", color="#2c7fb8")

    for i, (s, f) in enumerate(zip(setup, forward)):
        total = s + f
        ax.text(i, total + max(forward) * 0.02, f"{total:.1f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(libs, rotation=15, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Setup + Forward Cost ({int(max_dim)}^3 grid, C=32, K=3x3x3)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = "comparison_topology_overhead.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


# =============================================================================
# 4. Summary
# =============================================================================


def print_summary(df: pd.DataFrame) -> None:
    """Print a text summary of benchmark results."""
    print("\n" + "=" * 72)
    print("  Sparse Convolution Comparison Summary")
    print("=" * 72)

    libs = sorted(df["library"].unique())

    # Grid-size results
    grid_df = df[df["suite"] == "grid_size"]
    if not grid_df.empty:
        print("\n--- Grid-Size Scaling (C=32, K=3) ---")
        header = f"  {'Grid':>6s}  {'Voxels':>8s}"
        for lib in libs:
            header += f"  {lib:>16s}"
        print(header)
        for dim in sorted(grid_df["grid_dim"].unique()):
            vox = int(dim) ** 3
            row = f"  {int(dim):>4d}^3  {vox:>8d}"
            for lib in libs:
                sub = grid_df[(grid_df["grid_dim"] == dim) & (grid_df["library"] == lib)]
                if not sub.empty:
                    row += f"  {sub['mean_ms'].values[0]:>13.3f}ms"
                else:
                    row += f"  {'--':>16s}"
            print(row)

    # Sparsity results
    sparse_df = df[df["suite"] == "sparsity"]
    if not sparse_df.empty:
        for bbox_dim in sorted(sparse_df["bbox_dim"].unique()):
            bbox_sub = sparse_df[sparse_df["bbox_dim"] == bbox_dim]
            total = int(bbox_dim) ** 3
            print(f"\n--- Sparsity (bbox={int(bbox_dim)}, {total:,} cells, C=32) ---")
            header = f"  {'Occ%':>6s}  {'Voxels':>10s}"
            for lib in libs:
                header += f"  {lib:>16s}"
            print(header)
            for occ in sorted(bbox_sub["occupancy_pct"].unique()):
                occ_sub = bbox_sub[bbox_sub["occupancy_pct"] == occ]
                vox = occ_sub["num_voxels"].iloc[0] if not occ_sub.empty else 0
                row = f"  {int(occ):>5d}%  {int(vox):>10,d}"
                for lib in libs:
                    lsub = occ_sub[occ_sub["library"] == lib]
                    if not lsub.empty:
                        row += f"  {lsub['mean_ms'].values[0]:>13.3f}ms"
                    else:
                        row += f"  {'--':>16s}"
                print(row)

    # Channel scaling
    chan_df = df[df["suite"] == "channels"]
    if not chan_df.empty:
        print(f"\n--- Channel Scaling (16^3 grid, K=3) ---")
        header = f"  {'C':>6s}"
        for lib in libs:
            header += f"  {lib:>16s}"
        print(header)
        for c in sorted(chan_df["channels"].unique()):
            row = f"  {int(c):>6d}"
            for lib in libs:
                sub = chan_df[(chan_df["channels"] == c) & (chan_df["library"] == lib)]
                if not sub.empty:
                    row += f"  {sub['mean_ms'].values[0]:>13.3f}ms"
                else:
                    row += f"  {'--':>16s}"
            print(row)

    print()


# =============================================================================
# 5. Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize sparse convolution comparison benchmark results.",
    )
    parser.add_argument("--file", "-f", required=True, help="Path to comparison JSON results file")
    parser.add_argument(
        "--filter",
        nargs="*",
        default=None,
        help="Only show these libraries (name substrings, e.g. 'fVDB spconv')",
    )
    args = parser.parse_args()

    df = load_results(args.file)
    print(f"Loaded {len(df)} benchmark records.")

    if args.filter:
        mask = df["library"].apply(lambda lib: any(f.lower() in lib.lower() for f in args.filter))
        df = df[mask]
        print(f"Filtered to {len(df)} records for: {', '.join(args.filter)}")

    if df.empty:
        print("No data to plot.")
        sys.exit(0)

    print_summary(df)

    plot_grid_size_scaling(df)
    plot_sparsity_breakeven(df)
    plot_channel_scaling(df)
    plot_topology_overhead(df)


if __name__ == "__main__":
    main()
