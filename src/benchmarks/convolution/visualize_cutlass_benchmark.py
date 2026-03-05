# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# visualize_cutlass_benchmark.py -- Visualize CUTLASS vs GroupedGemm benchmark.
#
# Benchmark names follow the pattern:
#     BM_Cutlass_<Phase>_<Grid>_<Backend>/<dim>[/<Cin>/<Cout>]
#
# Phase: Topo, Fwd, Bwd
# Grid:  Dense, Sphere
# Backend: GroupedGemm (fp32), Cutlass (fp16)
#
# Figures:
#   1. Forward time vs channels (line plot, one panel per grid shape)
#   2. Backward time vs channels (same layout)
#   3. Speedup bar chart per phase (GroupedGemm_time / Cutlass_time)
#   4. Topology time comparison (bar chart)
#   5. Training iteration (forward + backward) time comparison
#
# Usage:
#   python visualize_cutlass_benchmark.py --file results/cutlass_benchmark.json
#   python visualize_cutlass_benchmark.py --run ./gbenchmarks/cutlass_benchmark

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BACKEND_LABELS = {
    "GroupedGemm": "GroupedGemm (fp32 cuBLAS)",
    "Cutlass": "CUTLASS (fp16 TensorCore)",
}

_BACKEND_COLORS = {
    "GroupedGemm": "#1f77b4",
    "Cutlass": "#ff7f0e",
}

_PHASE_LABELS = {
    "Topo": "Topology",
    "Fwd": "Forward",
    "Bwd": "Backward",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_benchmark_name(full_name):
    """Parse a Google Benchmark name into structured fields.

    Returns a dict with keys:
        Phase, Grid, Backend, Dim, Cin, Cout
    or None if the name does not match.
    """
    # BM_Cutlass_<Phase>_<Grid>_<Backend>/<dim>[/<Cin>/<Cout>]
    m = re.match(
        r"BM_Cutlass_(\w+?)_(Dense|Sphere)_(GroupedGemm|Cutlass)/(.+)", full_name
    )
    if not m:
        return None

    phase = m.group(1)
    grid = m.group(2)
    backend = m.group(3)
    args = m.group(4).split("/")

    result = {
        "Phase": phase,
        "Grid": grid,
        "Backend": backend,
        "Dim": int(args[0]),
    }

    if len(args) >= 3:
        result["Cin"] = int(args[1])
        result["Cout"] = int(args[2])
    else:
        result["Cin"] = 0
        result["Cout"] = 0

    return result


def load_results(path):
    """Load Google Benchmark JSON and return a pandas DataFrame."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for bm in data.get("benchmarks", []):
        name = bm.get("name", "")
        parsed = parse_benchmark_name(name)
        if parsed is None:
            continue

        # Skip aggregate rows (mean/median/stddev)
        run_type = bm.get("run_type", "iteration")
        if run_type != "iteration":
            continue

        parsed["time_ms"] = bm.get("real_time", 0.0)
        parsed["iterations"] = bm.get("iterations", 1)
        parsed["voxels"] = bm.get("Voxels", 0)
        parsed["channels"] = bm.get("Channels", 0)
        rows.append(parsed)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def _square_channel_label(cin, cout):
    if cin == cout:
        return f"C={cin}"
    return f"{cin}->{cout}"


# ---------------------------------------------------------------------------
# Figure 1 & 2: Time vs channels (forward / backward)
# ---------------------------------------------------------------------------


def plot_time_vs_channels(df, phase, out_dir):
    """Line plot: time vs Cin for square-channel cases, one panel per grid shape."""
    phase_df = df[(df["Phase"] == phase) & (df["Cin"] == df["Cout"])].copy()
    if phase_df.empty:
        return

    grids = sorted(phase_df["Grid"].unique())
    fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 5), squeeze=False)

    phase_label = _PHASE_LABELS.get(phase, phase)

    for col, grid in enumerate(grids):
        ax = axes[0, col]
        gdf = phase_df[phase_df["Grid"] == grid]

        # Use the largest grid size for a clean comparison
        max_dim = gdf["Dim"].max()
        gdf = gdf[gdf["Dim"] == max_dim]

        voxels = int(gdf["voxels"].iloc[0]) if len(gdf) > 0 else 0

        for backend in ["GroupedGemm", "Cutlass"]:
            bdf = gdf[gdf["Backend"] == backend].sort_values("Cin")
            if bdf.empty:
                continue
            ax.plot(
                bdf["Cin"],
                bdf["time_ms"],
                "o-",
                label=_BACKEND_LABELS.get(backend, backend),
                color=_BACKEND_COLORS.get(backend, None),
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Channels (Cin = Cout)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{phase_label} -- {grid} (dim={max_dim}, ~{voxels:,} voxels)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{phase_label} Time vs Channels: CUTLASS fp16 vs GroupedGemm fp32",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, f"cutlass_{phase.lower()}_time_vs_channels.png", out_dir)


# ---------------------------------------------------------------------------
# Figure 3: Speedup bar chart
# ---------------------------------------------------------------------------


def plot_speedup_bars(df, out_dir):
    """Bar chart: speedup (GroupedGemm / Cutlass) per config."""
    for phase in ["Fwd", "Bwd"]:
        phase_df = df[(df["Phase"] == phase) & (df["Cin"] == df["Cout"])].copy()
        if phase_df.empty:
            continue

        grids = sorted(phase_df["Grid"].unique())
        fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 5), squeeze=False)

        phase_label = _PHASE_LABELS.get(phase, phase)

        for col, grid in enumerate(grids):
            ax = axes[0, col]
            gdf = phase_df[phase_df["Grid"] == grid]

            dims = sorted(gdf["Dim"].unique())
            channels = sorted(gdf["Cin"].unique())

            x = np.arange(len(channels))
            width = 0.8 / max(len(dims), 1)

            for i, dim in enumerate(dims):
                speedups = []
                for cin in channels:
                    gg_row = gdf[(gdf["Dim"] == dim) & (gdf["Cin"] == cin) & (gdf["Backend"] == "GroupedGemm")]
                    cu_row = gdf[(gdf["Dim"] == dim) & (gdf["Cin"] == cin) & (gdf["Backend"] == "Cutlass")]
                    if len(gg_row) > 0 and len(cu_row) > 0:
                        gg_t = gg_row["time_ms"].values[0]
                        cu_t = cu_row["time_ms"].values[0]
                        speedups.append(gg_t / cu_t if cu_t > 0 else 0)
                    else:
                        speedups.append(0)

                offset = (i - len(dims) / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset,
                    speedups,
                    width,
                    label=f"dim={dim}",
                    alpha=0.85,
                )

                # Add value labels on bars
                for bar, sp in zip(bars, speedups):
                    if sp > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            f"{sp:.2f}x",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([f"C={c}" for c in channels])
            ax.set_ylabel("Speedup (GroupedGemm / CUTLASS)")
            ax.set_title(f"{phase_label} Speedup -- {grid}")
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"{phase_label} Speedup: CUTLASS fp16 vs GroupedGemm fp32 (>1 = CUTLASS faster)",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        _save(fig, f"cutlass_{phase.lower()}_speedup.png", out_dir)


# ---------------------------------------------------------------------------
# Figure 4: Topology time comparison
# ---------------------------------------------------------------------------


def plot_topology(df, out_dir):
    """Bar chart comparing topology construction time."""
    topo_df = df[df["Phase"] == "Topo"].copy()
    if topo_df.empty:
        return

    grids = sorted(topo_df["Grid"].unique())
    fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 5), squeeze=False)

    for col, grid in enumerate(grids):
        ax = axes[0, col]
        gdf = topo_df[topo_df["Grid"] == grid]

        dims = sorted(gdf["Dim"].unique())
        x = np.arange(len(dims))
        width = 0.35

        for i, backend in enumerate(["GroupedGemm", "Cutlass"]):
            times = []
            for dim in dims:
                row = gdf[(gdf["Dim"] == dim) & (gdf["Backend"] == backend)]
                times.append(row["time_ms"].values[0] if len(row) > 0 else 0)

            offset = (i - 0.5) * width
            ax.bar(
                x + offset,
                times,
                width,
                label=_BACKEND_LABELS.get(backend, backend),
                color=_BACKEND_COLORS.get(backend, None),
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("Grid dim / radius")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Topology -- {grid}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Topology Construction Time: CUTLASS GPU Builder vs GroupedGemm (dense kmap + CSR)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "cutlass_topology_time.png", out_dir)


# ---------------------------------------------------------------------------
# Figure 5: Training iteration (forward + backward)
# ---------------------------------------------------------------------------


def plot_training_iteration(df, out_dir):
    """Stacked bar: forward + backward time per config at largest grid size."""
    for grid in ["Dense", "Sphere"]:
        fwd_df = df[(df["Phase"] == "Fwd") & (df["Grid"] == grid) & (df["Cin"] == df["Cout"])].copy()
        bwd_df = df[(df["Phase"] == "Bwd") & (df["Grid"] == grid) & (df["Cin"] == df["Cout"])].copy()
        if fwd_df.empty or bwd_df.empty:
            continue

        max_dim = fwd_df["Dim"].max()
        fwd_df = fwd_df[fwd_df["Dim"] == max_dim]
        bwd_df = bwd_df[bwd_df["Dim"] == max_dim]

        channels = sorted(fwd_df["Cin"].unique())
        voxels = int(fwd_df["voxels"].iloc[0]) if len(fwd_df) > 0 else 0

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(channels))
        width = 0.35

        for i, backend in enumerate(["GroupedGemm", "Cutlass"]):
            fwd_times = []
            bwd_times = []
            for cin in channels:
                fr = fwd_df[(fwd_df["Cin"] == cin) & (fwd_df["Backend"] == backend)]
                br = bwd_df[(bwd_df["Cin"] == cin) & (bwd_df["Backend"] == backend)]
                fwd_times.append(fr["time_ms"].values[0] if len(fr) > 0 else 0)
                bwd_times.append(br["time_ms"].values[0] if len(br) > 0 else 0)

            offset = (i - 0.5) * width
            color = _BACKEND_COLORS.get(backend, None)
            ax.bar(
                x + offset,
                fwd_times,
                width,
                label=f"{_BACKEND_LABELS.get(backend, backend)} (fwd)",
                color=color,
                alpha=0.9,
            )
            ax.bar(
                x + offset,
                bwd_times,
                width,
                bottom=fwd_times,
                label=f"{_BACKEND_LABELS.get(backend, backend)} (bwd)",
                color=color,
                alpha=0.55,
                hatch="//",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"C={c}" for c in channels])
        ax.set_xlabel("Channels (Cin = Cout)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(
            f"Training Iteration (Fwd + Bwd) -- {grid} (dim={max_dim}, ~{voxels:,} voxels)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        _save(fig, f"cutlass_training_{grid.lower()}.png", out_dir)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def print_summary(df):
    """Print geometric mean speedups."""
    print("\n" + "=" * 70)
    print("CUTLASS vs GroupedGemm -- Summary (speedup = GG_time / CUTLASS_time)")
    print("=" * 70)

    for phase in ["Fwd", "Bwd"]:
        phase_df = df[(df["Phase"] == phase) & (df["Cin"] == df["Cout"])].copy()
        if phase_df.empty:
            continue

        speedups = []
        for grid in sorted(phase_df["Grid"].unique()):
            for dim in sorted(phase_df["Dim"].unique()):
                for cin in sorted(phase_df["Cin"].unique()):
                    gg = phase_df[
                        (phase_df["Grid"] == grid)
                        & (phase_df["Dim"] == dim)
                        & (phase_df["Cin"] == cin)
                        & (phase_df["Backend"] == "GroupedGemm")
                    ]
                    cu = phase_df[
                        (phase_df["Grid"] == grid)
                        & (phase_df["Dim"] == dim)
                        & (phase_df["Cin"] == cin)
                        & (phase_df["Backend"] == "Cutlass")
                    ]
                    if len(gg) > 0 and len(cu) > 0:
                        gg_t = gg["time_ms"].values[0]
                        cu_t = cu["time_ms"].values[0]
                        if cu_t > 0:
                            sp = gg_t / cu_t
                            speedups.append(sp)
                            print(
                                f"  {_PHASE_LABELS[phase]:>8s}  {grid:>6s}  dim={dim:<3d}  "
                                f"C={cin:<4d}  GG={gg_t:8.3f}ms  CU={cu_t:8.3f}ms  "
                                f"speedup={sp:.2f}x"
                            )

        if speedups:
            geo_mean = np.exp(np.mean(np.log(speedups)))
            print(
                f"\n  {_PHASE_LABELS[phase]} geometric mean speedup: {geo_mean:.2f}x"
            )
            print()

    # Topology summary
    topo_df = df[df["Phase"] == "Topo"].copy()
    if not topo_df.empty:
        print("Topology construction:")
        for grid in sorted(topo_df["Grid"].unique()):
            for dim in sorted(topo_df["Dim"].unique()):
                gg = topo_df[
                    (topo_df["Grid"] == grid)
                    & (topo_df["Dim"] == dim)
                    & (topo_df["Backend"] == "GroupedGemm")
                ]
                cu = topo_df[
                    (topo_df["Grid"] == grid)
                    & (topo_df["Dim"] == dim)
                    & (topo_df["Backend"] == "Cutlass")
                ]
                if len(gg) > 0 and len(cu) > 0:
                    gg_t = gg["time_ms"].values[0]
                    cu_t = cu["time_ms"].values[0]
                    sp = gg_t / cu_t if cu_t > 0 else 0
                    print(
                        f"  {grid:>6s}  dim={dim:<3d}  "
                        f"GG={gg_t:8.3f}ms  CU={cu_t:8.3f}ms  speedup={sp:.2f}x"
                    )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CUTLASS vs GroupedGemm benchmark results."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to Google Benchmark JSON output file")
    group.add_argument("--run", help="Path to benchmark executable (will run it)")
    parser.add_argument(
        "--filter", default="", help="Google Benchmark --benchmark_filter regex"
    )
    parser.add_argument(
        "--out-dir", default=".", help="Directory for output figures (default: cwd)"
    )
    args = parser.parse_args()

    if args.run:
        json_path = os.path.join(args.out_dir, "cutlass_benchmark.json")
        cmd = [
            args.run,
            "--benchmark_out_format=json",
            f"--benchmark_out={json_path}",
        ]
        if args.filter:
            cmd.append(f"--benchmark_filter={args.filter}")
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        args.file = json_path

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_results(args.file)
    if df.empty:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} benchmark results from {args.file}")

    # Generate all figures
    plot_time_vs_channels(df, "Fwd", args.out_dir)
    plot_time_vs_channels(df, "Bwd", args.out_dir)
    plot_speedup_bars(df, args.out_dir)
    plot_topology(df, args.out_dir)
    plot_training_iteration(df, args.out_dir)
    print_summary(df)


if __name__ == "__main__":
    main()
