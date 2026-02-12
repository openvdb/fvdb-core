# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# visualize_convolution_benchmark.py -- Visualize convolution benchmark results.
#
# Produces four figure groups:
#
#   1. Speedup heatmap:  new-backend speedup over old backend for every
#      (grid_pattern, grid_size, channel_count) combination.  This is the
#      single most useful view because it shows *where* each backend wins.
#
#   2. Time-vs-channels:  wall-clock time as a function of channel count for
#      each grid size, with all three backends overlaid.  Highlights the
#      compute-bound crossover where GEMM overtakes fused.
#
#   3. Time-vs-voxels:   wall-clock time as a function of voxel count for
#      each channel count, with all three backends overlaid.
#
#   4. Topology construction comparison (old pack-info conversion vs new
#      gather-scatter topology builder).
#
# The old visualize_benchmark.py bar charts only showed the comparison at the
# largest N, which buried the interesting small-C regime.  Here the heatmap
# and per-channel plots ensure every parameter combination is visible.
#

import argparse
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, TwoSlopeNorm

# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------

_BACKEND_LABELS = {
    "Old": "Old (KernelMap)",
    "GatherScatter": "GatherScatter (GEMM)",
    "GatherScatterFused": "GatherScatter (Fused)",
    "GroupedGemm": "CUTLASS GroupedGemm",
    "GatherScatter_LargeC": "GatherScatter (GEMM, large C)",
}

_BACKEND_ORDER = [
    "Old",
    "GatherScatter",
    "GatherScatter_LargeC",
    "GatherScatterFused",
    "GroupedGemm",
]


def parse_conv_benchmark_name(full_name):
    """
    Parse convolution benchmark names emitted by convolution_benchmark.cu.

    Convolution benchmarks:
        BM_Conv_<Grid>_<Backend>/<dim_or_radius>/<C_in>/<C_out>
    Topology benchmarks:
        BM_Topology_<Grid>_<Backend>/<dim_or_radius>

    Returns a dict of parsed fields.
    """
    name = full_name
    if name.startswith("BM_"):
        name = name[3:]

    parts = name.split("/")
    base = parts[0]
    # Google Benchmark appends "/real_time", "/manual_time", etc. -- drop non-numeric parts.
    args = [int(a) for a in parts[1:] if a.isdigit()]

    tokens = base.split("_")

    is_topology = tokens[0] == "Topology"

    # Grid pattern is the second token (Dense / Sphere)
    grid_pattern = tokens[1] if len(tokens) > 1 else "Unknown"

    # Backend is everything after Grid pattern
    backend = "_".join(tokens[2:]) if len(tokens) > 2 else "Unknown"

    if is_topology:
        dim = args[0] if len(args) > 0 else 0
        return {
            "Category": "Topology",
            "Grid": grid_pattern,
            "Backend": backend,
            "Dim": dim,
            "C_in": 0,
            "C_out": 0,
            "Name": base,
        }
    else:
        dim = args[0] if len(args) > 0 else 0
        c_in = args[1] if len(args) > 1 else 0
        c_out = args[2] if len(args) > 2 else 0
        return {
            "Category": "Convolution",
            "Grid": grid_pattern,
            "Backend": backend,
            "Dim": dim,
            "C_in": c_in,
            "C_out": c_out,
            "Name": base,
        }


def load_benchmarks(source):
    """Load Google Benchmark JSON and return a parsed DataFrame."""
    with open(source, "r") as f:
        data = json.load(f)

    records = []
    for b in data.get("benchmarks", []):
        if b.get("run_type") == "aggregate":
            continue
        info = parse_conv_benchmark_name(b["name"])

        time_unit = b.get("time_unit", "ns")
        real_time = b["real_time"]
        # Normalise to milliseconds
        if time_unit == "ns":
            time_ms = real_time / 1e6
        elif time_unit == "us":
            time_ms = real_time / 1e3
        elif time_unit == "ms":
            time_ms = real_time
        elif time_unit == "s":
            time_ms = real_time * 1e3
        else:
            time_ms = real_time

        info["Time_ms"] = time_ms
        info["Items"] = b.get("items_per_second", 0)
        records.append(info)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Execution
# ---------------------------------------------------------------------------


def run_benchmark(executable, output_file="convolution_results.json", filt=None):
    """Run the benchmark binary and write JSON output."""
    if not os.path.exists(executable) and os.path.exists("./" + executable):
        executable = "./" + executable

    cmd = [executable, f"--benchmark_out={output_file}", "--benchmark_out_format=json"]
    if filt:
        cmd.append(f"--benchmark_filter={filt}")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_file


# ---------------------------------------------------------------------------
# 3. Visualisation helpers
# ---------------------------------------------------------------------------


def _backend_color(backend):
    """Consistent colour per backend."""
    cmap = {
        "Old": "#d62728",                  # red
        "GatherScatter": "#1f77b4",        # blue
        "GatherScatterFused": "#2ca02c",   # green
        "GroupedGemm": "#ff7f0e",          # orange
        "GatherScatter_LargeC": "#9467bd", # purple
    }
    return cmap.get(backend, "#7f7f7f")


def _backend_label(backend):
    return _BACKEND_LABELS.get(backend, backend)


def _grid_label(grid, dim):
    if grid == "Dense":
        n = dim ** 3
        return f"Dense {dim}^3 ({n:,}v)"
    else:
        return f"Sphere R={dim}"


# ---------------------------------------------------------------------------
# 4. Plot: Speedup heatmap (the headline figure)
# ---------------------------------------------------------------------------


def plot_speedup_heatmap(df):
    """
    For each (Grid, Dim, C_in) show the speedup of GatherScatter and
    GatherScatterFused over the Old backend, as a pair of annotated heatmaps.

    This is the most important plot because it makes every parameter
    combination visible at once -- no single-N summary that hides the
    interesting small-C regime.
    """
    conv = df[df["Category"] == "Convolution"].copy()
    if conv.empty:
        return

    grids = conv["Grid"].unique()

    for grid in grids:
        gdata = conv[conv["Grid"] == grid]
        old = gdata[gdata["Backend"] == "Old"].copy()
        if old.empty:
            continue

        # Index by (Dim, C_in) for the old backend
        old_idx = old.set_index(["Dim", "C_in"])["Time_ms"]

        new_backends = [b for b in _BACKEND_ORDER if b != "Old" and b in gdata["Backend"].unique()]
        if not new_backends:
            continue

        fig, axes = plt.subplots(1, len(new_backends), figsize=(7 * len(new_backends), 5))
        if len(new_backends) == 1:
            axes = [axes]

        for ax, nb in zip(axes, new_backends):
            nb_data = gdata[gdata["Backend"] == nb].copy()
            rows = []
            for _, row in nb_data.iterrows():
                key = (row["Dim"], row["C_in"])
                if key in old_idx.index:
                    old_t = old_idx[key]
                    speedup = old_t / row["Time_ms"] if row["Time_ms"] > 0 else np.nan
                    rows.append({
                        "Grid Size": _grid_label(grid, row["Dim"]),
                        "Channels": f"C={row['C_in']}",
                        "Speedup": speedup,
                    })

            if not rows:
                continue

            hdf = pd.DataFrame(rows)
            pivot = hdf.pivot(index="Grid Size", columns="Channels", values="Speedup")

            # Sort columns by channel count numerically
            col_order = sorted(pivot.columns, key=lambda c: int(c.split("=")[1]))
            pivot = pivot[col_order]

            # Colour scale centred at 1x (white), <1 red, >1 green
            vmin = min(pivot.min().min(), 0.5)
            vmax = max(pivot.max().max(), 2.0)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                norm=norm,
                linewidths=0.5,
                ax=ax,
                cbar_kws={"label": "Speedup vs Old"},
            )
            ax.set_title(f"{_backend_label(nb)}\nspeedup over Old", fontsize=12)
            ax.set_ylabel("")
            ax.set_xlabel("")

        fig.suptitle(f"{grid} Grid -- Speedup over Old Backend", fontsize=14, y=1.02)
        plt.tight_layout()
        out = f"conv_speedup_heatmap_{grid.lower()}.png"
        print(f"  Saving {out}")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Plot: Time vs channel count (per grid size)
# ---------------------------------------------------------------------------


def plot_time_vs_channels(df):
    """
    For each grid pattern and grid size, plot wall-clock time vs channel
    count with all three backends overlaid.  This highlights the crossover
    where GEMM overtakes fused at high C.
    """
    conv = df[df["Category"] == "Convolution"].copy()
    if conv.empty:
        return

    grids = conv["Grid"].unique()

    for grid in grids:
        gdata = conv[conv["Grid"] == grid]
        dims = sorted(gdata["Dim"].unique())

        fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5), sharey=False)
        if len(dims) == 1:
            axes = [axes]

        for ax, dim in zip(axes, dims):
            subset = gdata[gdata["Dim"] == dim]
            for backend in _BACKEND_ORDER:
                bdata = subset[subset["Backend"] == backend].sort_values("C_in")
                if bdata.empty:
                    continue
                ax.plot(
                    bdata["C_in"],
                    bdata["Time_ms"],
                    marker="o",
                    linewidth=2,
                    markersize=7,
                    label=_backend_label(backend),
                    color=_backend_color(backend),
                )

            ax.set_xlabel("Channel count (C_in = C_out)")
            ax.set_ylabel("Time (ms)")
            ax.set_title(_grid_label(grid, dim))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(sorted(subset["C_in"].unique()))

        fig.suptitle(f"{grid} Grid -- Time vs Channel Count", fontsize=14, y=1.02)
        plt.tight_layout()
        out = f"conv_time_vs_channels_{grid.lower()}.png"
        print(f"  Saving {out}")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Plot: Time vs voxel count (per channel count)
# ---------------------------------------------------------------------------


def plot_time_vs_voxels(df):
    """
    For each grid pattern and channel count, plot wall-clock time vs
    grid size with all three backends overlaid.
    """
    conv = df[df["Category"] == "Convolution"].copy()
    if conv.empty:
        return

    grids = conv["Grid"].unique()

    for grid in grids:
        gdata = conv[conv["Grid"] == grid]
        channels = sorted(gdata["C_in"].unique())

        fig, axes = plt.subplots(1, len(channels), figsize=(6 * len(channels), 5), sharey=False)
        if len(channels) == 1:
            axes = [axes]

        for ax, ch in zip(axes, channels):
            subset = gdata[gdata["C_in"] == ch]
            for backend in _BACKEND_ORDER:
                bdata = subset[subset["Backend"] == backend].sort_values("Dim")
                if bdata.empty:
                    continue

                # Compute approximate voxel counts for x-axis
                if grid == "Dense":
                    voxels = bdata["Dim"].values ** 3
                else:
                    # Rough sphere shell count
                    voxels = (4 * np.pi * bdata["Dim"].values ** 2).astype(int)

                ax.plot(
                    voxels,
                    bdata["Time_ms"].values,
                    marker="o",
                    linewidth=2,
                    markersize=7,
                    label=_backend_label(backend),
                    color=_backend_color(backend),
                )

            ax.set_xlabel("Approx. voxel count")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"C_in = C_out = {ch}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

        fig.suptitle(f"{grid} Grid -- Time vs Voxel Count", fontsize=14, y=1.02)
        plt.tight_layout()
        out = f"conv_time_vs_voxels_{grid.lower()}.png"
        print(f"  Saving {out}")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Plot: Topology construction comparison
# ---------------------------------------------------------------------------


def plot_topology(df):
    """Bar chart comparing old vs new topology construction time."""
    topo = df[df["Category"] == "Topology"].copy()
    if topo.empty:
        print("  (No topology benchmarks found, skipping.)")
        return

    grids = topo["Grid"].unique()

    for grid in grids:
        gdata = topo[topo["Grid"] == grid]
        dims = sorted(gdata["Dim"].unique())

        fig, ax = plt.subplots(figsize=(8, 5))

        backends = gdata["Backend"].unique()
        x = np.arange(len(dims))
        width = 0.8 / max(len(backends), 1)

        for i, backend in enumerate(backends):
            bdata = gdata[gdata["Backend"] == backend]
            times = []
            for dim in dims:
                t = bdata[bdata["Dim"] == dim]["Time_ms"]
                times.append(t.values[0] if len(t) > 0 else 0)

            offset = (i - len(backends) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                times,
                width,
                label=_backend_label(backend),
                color=_backend_color(backend),
            )
            for bar, val in zip(bars, times):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([_grid_label(grid, d) for d in dims], fontsize=9)
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Topology Construction -- {grid} Grid")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = f"conv_topology_{grid.lower()}.png"
        print(f"  Saving {out}")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Plot: Per-channel-count grouped bar chart (replaces misleading max-N-only)
# ---------------------------------------------------------------------------


def plot_grouped_bars_by_channel(df):
    """
    Instead of a single bar chart at the largest grid, show a grouped-bar
    comparison at EVERY channel count, using the median grid size.  This
    avoids the trap of summarising only at the largest or smallest extreme.
    """
    conv = df[df["Category"] == "Convolution"].copy()
    if conv.empty:
        return

    grids = conv["Grid"].unique()

    for grid in grids:
        gdata = conv[conv["Grid"] == grid]
        channels = sorted(gdata["C_in"].unique())
        # Use the middle grid size as a representative
        dims = sorted(gdata["Dim"].unique())
        mid_dim = dims[len(dims) // 2] if dims else 0

        subset = gdata[gdata["Dim"] == mid_dim]
        if subset.empty:
            continue

        backends = [b for b in _BACKEND_ORDER if b in subset["Backend"].unique()]

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(channels))
        width = 0.8 / max(len(backends), 1)

        for i, backend in enumerate(backends):
            bdata = subset[subset["Backend"] == backend]
            times = []
            for ch in channels:
                t = bdata[bdata["C_in"] == ch]["Time_ms"]
                times.append(t.values[0] if len(t) > 0 else 0)

            offset = (i - len(backends) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                times,
                width,
                label=_backend_label(backend),
                color=_backend_color(backend),
            )
            for bar, val in zip(bars, times):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([f"C={ch}" for ch in channels])
        ax.set_ylabel("Time (ms)")
        ax.set_title(
            f"{grid} Grid, {_grid_label(grid, mid_dim)} -- "
            f"Backend Comparison by Channel Count"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = f"conv_bars_by_channel_{grid.lower()}.png"
        print(f"  Saving {out}")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Text summary
# ---------------------------------------------------------------------------


def print_summary(df):
    """Print a concise text summary of results."""
    conv = df[df["Category"] == "Convolution"].copy()
    if conv.empty:
        return

    print("\n=== Convolution Benchmark Summary ===\n")

    grids = conv["Grid"].unique()
    for grid in grids:
        gdata = conv[conv["Grid"] == grid]
        print(f"--- {grid} Grid ---")

        old = gdata[gdata["Backend"] == "Old"].set_index(["Dim", "C_in"])["Time_ms"]
        for backend in ["GatherScatter", "GatherScatterFused"]:
            bdata = gdata[gdata["Backend"] == backend]
            if bdata.empty:
                continue

            speedups = []
            for _, row in bdata.iterrows():
                key = (row["Dim"], row["C_in"])
                if key in old.index:
                    su = old[key] / row["Time_ms"] if row["Time_ms"] > 0 else float("nan")
                    speedups.append(su)
                    dim_label = _grid_label(grid, row["Dim"])
                    faster = "FASTER" if su > 1 else "slower"
                    print(
                        f"  {_backend_label(backend):30s}  "
                        f"{dim_label:25s}  C={row['C_in']:3d}  "
                        f"{row['Time_ms']:8.3f} ms  "
                        f"{su:5.2f}x ({faster})"
                    )

            if speedups:
                geo_mean = np.exp(np.mean(np.log([s for s in speedups if s > 0])))
                print(f"  {'':30s}  {'Geometric mean speedup:':25s}         {geo_mean:.2f}x")
            print()

    # Topology
    topo = df[df["Category"] == "Topology"]
    if not topo.empty:
        print("--- Topology Construction ---")
        for _, row in topo.iterrows():
            label = _grid_label(row["Grid"], row["Dim"])
            print(
                f"  {_backend_label(row['Backend']):30s}  "
                f"{label:25s}  {row['Time_ms']:8.3f} ms"
            )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize convolution benchmark results."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=str, help="Path to convolution_benchmark executable")
    group.add_argument("--file", type=str, help="Path to existing JSON results")
    parser.add_argument("--filter", type=str, default=None, help="Regex benchmark filter")
    args = parser.parse_args()

    if args.run:
        json_path = run_benchmark(args.run, filt=args.filter)
    else:
        json_path = args.file

    df = load_benchmarks(json_path)

    if args.filter and not args.run:
        df = df[df["Name"].str.contains(args.filter, regex=True)]

    print(f"Loaded {len(df)} benchmark records.\n")
    if df.empty:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")

    print_summary(df)

    print("Generating plots...")
    plot_speedup_heatmap(df)
    plot_time_vs_channels(df)
    plot_time_vs_voxels(df)
    plot_grouped_bars_by_channel(df)
    plot_topology(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
