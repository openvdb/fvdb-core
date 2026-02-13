# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# visualize_convolution_benchmark.py -- Visualize convolution benchmark results.
#
# Benchmark names follow the pattern:
#     BM_Conv_<Grid>_<Backend>[_CPU]/<dim>/<C_in>/<C_out>
#     BM_Topology_<Grid>_<Backend>/<dim>
#
# The optional _CPU suffix selects the device; without it, CUDA is assumed.
# CUDA and CPU results are plotted on separate figures because the absolute
# time scales differ by orders of magnitude.
#
# Figure groups:
#   Per device (CUDA, CPU):
#     1. Time-vs-channels line plot (per grid size, all backends overlaid)
#     2. Grouped bar chart at the median grid size
#     3. Speedup-vs-Old bar chart (GatherScatter, GatherScatterFused, GroupedGemm)
#   CUDA only:
#     4. Speedup heatmap (each backend vs Old)
#     5. Time-vs-voxels line plot
#     6. Topology construction bar chart

import argparse
import json
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BACKEND_LABELS = {
    "Old": "Old (KernelMap)",
    "GatherScatter": "GatherScatter (GEMM)",
    "GatherScatterFused": "GatherScatter (Fused)",
    "GroupedGemm": "GroupedGemm (Compacted)",
}

_BACKEND_ORDER = ["Old", "GatherScatter", "GatherScatterFused", "GroupedGemm"]

_BACKEND_COLORS = {
    "Old": "#d62728",
    "GatherScatter": "#1f77b4",
    "GatherScatterFused": "#2ca02c",
    "GroupedGemm": "#ff7f0e",
}

# Backends to compare head-to-head against Old
_COMPARE_BACKENDS = ["GatherScatter", "GatherScatterFused", "GroupedGemm"]

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Known backends (longest first so greedy match works)
_KNOWN_BACKENDS = sorted(
    ["Old", "GatherScatter", "GatherScatterFused", "GroupedGemm"],
    key=len,
    reverse=True,
)


def parse_benchmark_name(full_name):
    """Parse a Google Benchmark name into structured fields.

    Returns a dict with keys:
        Category, Grid, Backend, Device, Dim, C_in, C_out
    """
    name = full_name
    if name.startswith("BM_"):
        name = name[3:]

    # Split off numeric args and trailing /real_time etc.
    parts = name.split("/")
    base = parts[0]
    args = [int(p) for p in parts[1:] if p.isdigit()]

    # Detect category
    if base.startswith("Topology_"):
        category = "Topology"
        rest = base[len("Topology_"):]
    elif base.startswith("Conv_"):
        category = "Convolution"
        rest = base[len("Conv_"):]
    else:
        return None

    # Detect device suffix
    device = "CUDA"
    if rest.endswith("_CPU"):
        device = "CPU"
        rest = rest[: -len("_CPU")]

    # Grid is the first token
    underscore = rest.find("_")
    if underscore < 0:
        return None
    grid = rest[:underscore]
    backend_str = rest[underscore + 1:]

    # Match backend against known names (longest first)
    backend = backend_str
    for known in _KNOWN_BACKENDS:
        if backend_str == known:
            backend = known
            break

    dim = args[0] if len(args) > 0 else 0
    c_in = args[1] if len(args) > 1 else 0
    c_out = args[2] if len(args) > 2 else 0

    return {
        "Category": category,
        "Grid": grid,
        "Backend": backend,
        "Device": device,
        "Dim": dim,
        "C_in": c_in,
        "C_out": c_out,
    }


def load_benchmarks(source):
    """Load Google Benchmark JSON and return a parsed DataFrame."""
    with open(source, "r") as f:
        data = json.load(f)

    records = []
    for b in data.get("benchmarks", []):
        if b.get("run_type") == "aggregate":
            continue
        info = parse_benchmark_name(b["name"])
        if info is None:
            continue

        time_unit = b.get("time_unit", "ns")
        real_time = b["real_time"]
        scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}.get(time_unit, 1.0)
        info["Time_ms"] = real_time * scale
        info["Items"] = b.get("items_per_second", 0)

        # Pull custom counters emitted by the benchmark (optional)
        info["Voxels_counter"] = b.get("Voxels", 0)
        info["Channels_counter"] = b.get("Channels", 0)

        records.append(info)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Execution
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
# Helpers
# ---------------------------------------------------------------------------


def _color(backend):
    return _BACKEND_COLORS.get(backend, "#7f7f7f")


def _label(backend):
    return _BACKEND_LABELS.get(backend, backend)


def _grid_label(grid, dim):
    if grid == "Dense":
        return f"Dense {dim}^3 ({dim ** 3:,}v)"
    return f"Sphere R={dim}"


def _save(fig, name):
    print(f"  Saving {name}")
    fig.savefig(name, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _approx_voxels(grid, dim):
    """Approximate voxel count from grid type and dimension."""
    if grid == "Dense":
        return dim ** 3
    return int(4 * np.pi * dim ** 2)


# ---------------------------------------------------------------------------
# Plot: Time vs channels (line plot, one subplot per grid size)
# ---------------------------------------------------------------------------


def plot_time_vs_channels(df, device, suffix):
    """Line plot: time vs channel count, per grid size, all backends overlaid."""
    conv = df[(df["Category"] == "Convolution") & (df["Device"] == device)]
    if conv.empty:
        return

    for grid in sorted(conv["Grid"].unique()):
        gdata = conv[conv["Grid"] == grid]
        dims = sorted(gdata["Dim"].unique())
        if not dims:
            continue

        ncols = len(dims)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5), squeeze=False)

        for ax, dim in zip(axes[0], dims):
            subset = gdata[gdata["Dim"] == dim]
            for backend in _BACKEND_ORDER:
                bd = subset[subset["Backend"] == backend].sort_values("C_in")
                if bd.empty:
                    continue
                ax.plot(
                    bd["C_in"], bd["Time_ms"],
                    marker="o", linewidth=2, markersize=6,
                    label=_label(backend), color=_color(backend),
                )
            ax.set_xlabel("Channels (C_in = C_out)")
            ax.set_ylabel("Time (ms)")
            ax.set_title(_grid_label(grid, dim), fontsize=11)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)
            xticks = sorted(subset["C_in"].unique())
            if xticks:
                ax.set_xticks(xticks)

        fig.suptitle(f"{grid} Grid -- Time vs Channels [{device}]", fontsize=13, y=1.02)
        fig.tight_layout()
        _save(fig, f"conv_time_vs_channels_{grid.lower()}_{suffix}.png")


# ---------------------------------------------------------------------------
# Plot: Grouped bars at median grid size
# ---------------------------------------------------------------------------


def plot_bars_by_channel(df, device, suffix):
    """Grouped bar chart comparing backends at the median grid size."""
    conv = df[(df["Category"] == "Convolution") & (df["Device"] == device)]
    if conv.empty:
        return

    for grid in sorted(conv["Grid"].unique()):
        gdata = conv[conv["Grid"] == grid]
        dims = sorted(gdata["Dim"].unique())
        if not dims:
            continue
        mid_dim = dims[len(dims) // 2]
        subset = gdata[gdata["Dim"] == mid_dim]
        channels = sorted(subset["C_in"].unique())
        backends = [b for b in _BACKEND_ORDER if b in subset["Backend"].values]
        if not backends or not channels:
            continue

        fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(channels)), 5))
        x = np.arange(len(channels))
        w = 0.8 / len(backends)

        for i, backend in enumerate(backends):
            bd = subset[subset["Backend"] == backend]
            times = [bd[bd["C_in"] == c]["Time_ms"].values for c in channels]
            times = [t[0] if len(t) else 0.0 for t in times]
            offset = (i - len(backends) / 2 + 0.5) * w
            bars = ax.bar(x + offset, times, w, label=_label(backend), color=_color(backend))
            for bar, val in zip(bars, times):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"C={c}" for c in channels])
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{grid} {_grid_label(grid, mid_dim)} -- Backend Comparison [{device}]")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, f"conv_bars_{grid.lower()}_{suffix}.png")


# ---------------------------------------------------------------------------
# Plot: Speedup-vs-Old bar chart (focused comparison)
# ---------------------------------------------------------------------------


def plot_speedup_bars(df, device, suffix):
    """Bar chart: speedup of each backend over Old, per grid size."""
    conv = df[(df["Category"] == "Convolution") & (df["Device"] == device)]
    if conv.empty:
        return

    for grid in sorted(conv["Grid"].unique()):
        gdata = conv[conv["Grid"] == grid]
        old = gdata[gdata["Backend"] == "Old"]
        if old.empty:
            continue
        old_idx = old.set_index(["Dim", "C_in"])["Time_ms"]

        dims = sorted(gdata["Dim"].unique())
        channels = sorted(gdata["C_in"].unique())
        compare = [b for b in _COMPARE_BACKENDS if b in gdata["Backend"].values]
        if not compare or not channels:
            continue

        ncols = len(dims)
        fig, axes = plt.subplots(1, ncols, figsize=(max(5, 2 * len(channels)) * ncols, 5), squeeze=False)

        for ax, dim in zip(axes[0], dims):
            x = np.arange(len(channels))
            w = 0.8 / len(compare)

            for i, backend in enumerate(compare):
                bd = gdata[(gdata["Backend"] == backend) & (gdata["Dim"] == dim)]
                speedups = []
                for ch in channels:
                    row = bd[bd["C_in"] == ch]
                    key = (dim, ch)
                    if not row.empty and key in old_idx.index and row.iloc[0]["Time_ms"] > 0:
                        speedups.append(old_idx[key] / row.iloc[0]["Time_ms"])
                    else:
                        speedups.append(0.0)

                offset = (i - len(compare) / 2 + 0.5) * w
                bars = ax.bar(x + offset, speedups, w, label=_label(backend), color=_color(backend))
                for bar, val in zip(bars, speedups):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{val:.2f}x",
                            ha="center", va="bottom", fontsize=7,
                        )

            ax.axhline(y=1.0, color=_color("Old"), linestyle="--", linewidth=1.5, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"C={c}" for c in channels])
            ax.set_ylabel("Speedup over Old")
            ax.set_title(f"{_grid_label(grid, dim)}", fontsize=11)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{grid} Grid -- Speedup over Old [{device}]", fontsize=13, y=1.02)
        fig.tight_layout()
        _save(fig, f"conv_speedup_bars_{grid.lower()}_{suffix}.png")


# ---------------------------------------------------------------------------
# Plot: Speedup heatmap (CUDA only, each backend vs Old)
# ---------------------------------------------------------------------------


def plot_speedup_heatmap(df):
    """Heatmap of speedup over Old for every (grid_size, channel) pair."""
    conv = df[(df["Category"] == "Convolution") & (df["Device"] == "CUDA")]
    if conv.empty:
        return

    for grid in sorted(conv["Grid"].unique()):
        gdata = conv[conv["Grid"] == grid]
        old = gdata[gdata["Backend"] == "Old"]
        if old.empty:
            continue
        old_idx = old.set_index(["Dim", "C_in"])["Time_ms"]

        new_backends = [b for b in _BACKEND_ORDER
                        if b != "Old" and b in gdata["Backend"].values]
        if not new_backends:
            continue

        ncols = len(new_backends)
        fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5), squeeze=False)

        for ax, nb in zip(axes[0], new_backends):
            nbd = gdata[gdata["Backend"] == nb]
            rows = []
            for _, row in nbd.iterrows():
                key = (row["Dim"], row["C_in"])
                if key in old_idx.index and row["Time_ms"] > 0:
                    rows.append({
                        "Grid Size": _grid_label(grid, row["Dim"]),
                        "Channels": f"C={int(row['C_in'])}",
                        "Speedup": old_idx[key] / row["Time_ms"],
                    })
            if not rows:
                continue

            hdf = pd.DataFrame(rows)
            pivot = hdf.pivot(index="Grid Size", columns="Channels", values="Speedup")
            col_order = sorted(pivot.columns, key=lambda c: int(c.split("=")[1]))
            pivot = pivot[col_order]

            vmin = min(pivot.min().min(), 0.5)
            vmax = max(pivot.max().max(), 2.0)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", norm=norm,
                        linewidths=0.5, ax=ax, cbar_kws={"label": "Speedup vs Old"})
            ax.set_title(f"{_label(nb)}\nspeedup over Old", fontsize=11)
            ax.set_ylabel("")
            ax.set_xlabel("")

        fig.suptitle(f"{grid} Grid -- Speedup over Old [CUDA]", fontsize=13, y=1.02)
        fig.tight_layout()
        _save(fig, f"conv_speedup_heatmap_{grid.lower()}.png")


# ---------------------------------------------------------------------------
# Plot: Time vs voxel count (CUDA only)
# ---------------------------------------------------------------------------


def plot_time_vs_voxels(df):
    """Line plot: time vs approximate voxel count, per channel count."""
    conv = df[(df["Category"] == "Convolution") & (df["Device"] == "CUDA")]
    if conv.empty:
        return

    for grid in sorted(conv["Grid"].unique()):
        gdata = conv[conv["Grid"] == grid]
        channels = sorted(gdata["C_in"].unique())
        if not channels:
            continue

        ncols = len(channels)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5), squeeze=False)

        for ax, ch in zip(axes[0], channels):
            subset = gdata[gdata["C_in"] == ch]
            for backend in _BACKEND_ORDER:
                bd = subset[subset["Backend"] == backend].sort_values("Dim")
                if bd.empty:
                    continue
                voxels = np.array([_approx_voxels(grid, d) for d in bd["Dim"].values])
                ax.plot(voxels, bd["Time_ms"].values,
                        marker="o", linewidth=2, markersize=6,
                        label=_label(backend), color=_color(backend))
            ax.set_xlabel("Approx. voxel count")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"C={int(ch)}", fontsize=11)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)
            if len(subset["Dim"].unique()) > 1:
                ax.set_xscale("log")

        fig.suptitle(f"{grid} Grid -- Time vs Voxels [CUDA]", fontsize=13, y=1.02)
        fig.tight_layout()
        _save(fig, f"conv_time_vs_voxels_{grid.lower()}.png")


# ---------------------------------------------------------------------------
# Plot: Topology construction
# ---------------------------------------------------------------------------


def plot_topology(df):
    """Bar chart comparing topology construction backends."""
    topo = df[(df["Category"] == "Topology") & (df["Device"] == "CUDA")]
    if topo.empty:
        print("  (No topology benchmarks found, skipping.)")
        return

    for grid in sorted(topo["Grid"].unique()):
        gdata = topo[topo["Grid"] == grid]
        dims = sorted(gdata["Dim"].unique())
        backends = [b for b in _BACKEND_ORDER if b in gdata["Backend"].values]
        if not backends:
            continue

        fig, ax = plt.subplots(figsize=(max(8, 2 * len(dims)), 5))
        x = np.arange(len(dims))
        w = 0.8 / max(len(backends), 1)

        for i, backend in enumerate(backends):
            bd = gdata[gdata["Backend"] == backend]
            times = []
            for dim in dims:
                t = bd[bd["Dim"] == dim]["Time_ms"]
                times.append(t.values[0] if len(t) > 0 else 0.0)
            offset = (i - len(backends) / 2 + 0.5) * w
            bars = ax.bar(x + offset, times, w, label=_label(backend), color=_color(backend))
            for bar, val in zip(bars, times):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([_grid_label(grid, d) for d in dims], fontsize=9)
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Topology Construction -- {grid} Grid [CUDA]")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, f"conv_topology_{grid.lower()}.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

_SEP = "=" * 80


def _print_speedup_table(gdata, grid, old_idx, backends):
    """Print a per-backend speedup table relative to Old."""
    for backend in backends:
        bd = gdata[gdata["Backend"] == backend]
        if bd.empty:
            continue

        speedups = []
        lines = []
        for _, row in bd.sort_values(["Dim", "C_in"]).iterrows():
            key = (row["Dim"], row["C_in"])
            if key in old_idx.index and row["Time_ms"] > 0:
                su = old_idx[key] / row["Time_ms"]
                speedups.append(su)
                tag = "FASTER" if su > 1 else "slower"
                lines.append(
                    f"      {_grid_label(grid, row['Dim']):25s}  "
                    f"C={int(row['C_in']):3d}  "
                    f"{row['Time_ms']:8.3f} ms  {su:5.2f}x ({tag})"
                )

        print(f"    {_label(backend)}:")
        for ln in lines:
            print(ln)
        if speedups:
            geo = np.exp(np.mean(np.log(speedups)))
            fastest = max(speedups)
            slowest = min(speedups)
            print(
                f"      {'Geo-mean:':25s}         "
                f"       {geo:5.2f}x   "
                f"(range {slowest:.2f}x -- {fastest:.2f}x)"
            )
        print()


def print_summary(df):
    """Print a concise text summary of results."""
    for device in sorted(df["Device"].unique()):
        conv = df[(df["Category"] == "Convolution") & (df["Device"] == device)]
        if conv.empty:
            continue

        print(f"\n{_SEP}")
        print(f"  {device} Convolution Summary")
        print(f"{_SEP}\n")

        for grid in sorted(conv["Grid"].unique()):
            gdata = conv[conv["Grid"] == grid]
            print(f"  --- {grid} Grid ---\n")

            old = gdata[gdata["Backend"] == "Old"]
            if not old.empty:
                old_idx = old.set_index(["Dim", "C_in"])["Time_ms"]
                _print_speedup_table(gdata, grid, old_idx,
                                     [b for b in _BACKEND_ORDER if b != "Old"])
            else:
                # No Old baseline -- just print raw times
                for backend in _BACKEND_ORDER:
                    bd = gdata[gdata["Backend"] == backend]
                    for _, row in bd.sort_values(["Dim", "C_in"]).iterrows():
                        print(
                            f"    {_label(backend):32s}  "
                            f"{_grid_label(grid, row['Dim']):25s}  "
                            f"C={int(row['C_in']):3d}  {row['Time_ms']:8.3f} ms"
                        )
                print()

    # Topology
    topo = df[df["Category"] == "Topology"]
    if not topo.empty:
        print(f"\n{_SEP}")
        print("  Topology Construction Summary")
        print(f"{_SEP}\n")
        for grid in sorted(topo["Grid"].unique()):
            gtopo = topo[topo["Grid"] == grid]
            print(f"  --- {grid} Grid ---\n")
            for _, row in gtopo.sort_values(["Backend", "Dim"]).iterrows():
                print(
                    f"    {_label(row['Backend']):32s}  "
                    f"{_grid_label(row['Grid'], row['Dim']):25s}  "
                    f"{row['Time_ms']:8.3f} ms"
                )
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize convolution benchmark results.")
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
        df = df[df["Backend"].str.contains(args.filter, regex=True, na=False)]

    print(f"Loaded {len(df)} benchmark records.\n")
    if df.empty:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")

    print_summary(df)

    print("Generating plots...")

    # Per-device plots
    for device in sorted(df["Device"].unique()):
        suffix = device.lower()
        plot_time_vs_channels(df, device, suffix)
        plot_bars_by_channel(df, device, suffix)
        plot_speedup_bars(df, device, suffix)

    # CUDA-only plots
    plot_speedup_heatmap(df)
    plot_time_vs_voxels(df)
    plot_topology(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
