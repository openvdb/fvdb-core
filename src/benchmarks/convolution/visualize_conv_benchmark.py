# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Visualization for GatherScatterDefault sparse convolution benchmarks.
#
# Usage:
#   python visualize_conv_benchmark.py --run ./gather_scatter_conv_benchmark
#   python visualize_conv_benchmark.py --file results/gather_scatter_conv_benchmark.json
#   python visualize_conv_benchmark.py --file results.json --filter "CUDA"
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
from matplotlib.ticker import FuncFormatter

# =============================================================================
# 1. Parsing
# =============================================================================


def parse_conv_benchmark_name(full_name):
    """
    Parse convolution benchmark names.

    Formats:
        BM_Conv_<Group>_<Device>[_C<channels>]/<Param>
        BM_Conv_DenseRef_<Device>_C<channels>

    Examples:
        BM_Conv_Forward_CUDA_C32/16
        BM_Conv_Sparsity_CUDA_C32/25
        BM_Conv_ChannelScale_CUDA/64
        BM_Conv_TopologyBuild_CPU/32
        BM_Conv_DenseRef_CUDA_C32
    """
    if full_name.startswith("BM_"):
        full_name = full_name[3:]

    # Google Benchmark appends /real_time when UseRealTime() is set.
    # Split on "/" and collect all numeric parts as positional parameters.
    parts = full_name.split("/")
    base_name = parts[0]
    numeric_params = [int(p) for p in parts[1:] if p.isdigit()]
    param = numeric_params[0] if len(numeric_params) >= 1 else 0
    param2 = numeric_params[1] if len(numeric_params) >= 2 else 0

    tokens = base_name.split("_")

    # tokens[0] == "Conv", skip it
    tokens = tokens[1:]

    # Extract device
    device = "Unknown"
    device_idx = -1
    for i, tok in enumerate(tokens):
        if tok in ("CUDA", "CPU"):
            device = tok
            device_idx = i
            break

    # Group is everything before device
    group = "_".join(tokens[:device_idx]) if device_idx > 0 else tokens[0]

    # Channels: look for C<number> after device
    channels = 0
    for tok in tokens[device_idx + 1 :]:
        if tok.startswith("C") and tok[1:].isdigit():
            channels = int(tok[1:])

    return {
        "Group": group,
        "Device": device,
        "Channels": channels,
        "Param": param,
        "Param2": param2,
        "Name": base_name,
    }


def load_benchmarks(source, is_file=True):
    """Load benchmark JSON and parse into a DataFrame."""
    try:
        if is_file:
            with open(source, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(source)

        benchmarks = data.get("benchmarks", [])

        records = []
        for b in benchmarks:
            if b.get("run_type") == "aggregate":
                continue

            info = parse_conv_benchmark_name(b["name"])

            time_unit = b.get("time_unit", "ns")
            real_time = b["real_time"]
            if time_unit == "ms":
                time_ms = real_time
            elif time_unit == "us":
                time_ms = real_time / 1e3
            else:
                time_ms = real_time / 1e6

            info["Time (ms)"] = time_ms
            info["Real Time (ns)"] = b["real_time"]

            for key in ("voxels", "pairs", "occupancy", "channels"):
                if key in b:
                    info[key] = b[key]
                elif key not in info:
                    info[key] = 0

            throughput = b.get("items_per_second", 0)
            info["Throughput"] = throughput

            records.append(info)

        return pd.DataFrame(records)

    except FileNotFoundError:
        print(f"Error: File {source} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {source}.")
        sys.exit(1)


# =============================================================================
# 2. Execution
# =============================================================================


def run_benchmark_executable(executable_path, output_file="conv_results.json", benchmark_filter=None):
    """Run the C++ benchmark binary and save JSON output."""
    if not os.path.exists(executable_path) and not executable_path.startswith("./"):
        if os.path.exists("./" + executable_path):
            executable_path = "./" + executable_path

    print(f"Running benchmark: {executable_path}")
    print(f"Outputting to: {output_file}")

    cmd = [executable_path, f"--benchmark_out={output_file}", "--benchmark_out_format=json"]

    if benchmark_filter:
        cmd.append(f"--benchmark_filter={benchmark_filter}")
        print(f"Filtering benchmarks: {benchmark_filter}")

    try:
        subprocess.run(cmd, check=True)
        print("Benchmark run complete.")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Executable not found: {executable_path}")
        sys.exit(1)


# =============================================================================
# 3. Axis formatters
# =============================================================================


def format_voxels(x, pos):
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    elif x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def format_ms(x, pos):
    if x >= 1000:
        return f"{x / 1000:.1f}s"
    elif x >= 1:
        return f"{x:.1f}ms"
    elif x >= 0.001:
        return f"{x * 1000:.0f}us"
    return f"{x:.4f}ms"


# =============================================================================
# 4. Plot functions
# =============================================================================


def plot_throughput_vs_grid_size(df):
    """
    Plot 1: Time vs grid size for Forward, Backward, DenseBaseline, TopologyBuild.
    Separate panels per device.
    """
    groups = ["Forward", "Backward", "DenseBaseline", "TopologyBuild"]
    plot_df = df[df["Group"].isin(groups) & (df["Param"] > 0)].copy()

    if plot_df.empty:
        print("No data for throughput vs grid size plot.")
        return

    plot_df["Voxels"] = plot_df["Param"].apply(lambda d: d**3)

    sns.set_theme(style="whitegrid", context="talk")
    devices = sorted(plot_df["Device"].unique())

    palette = sns.color_palette("husl", len(groups))
    color_map = dict(zip(groups, palette))
    style_map = {"Forward": "-", "Backward": "--", "DenseBaseline": ":", "TopologyBuild": "-."}

    for device in devices:
        dev_data = plot_df[plot_df["Device"] == device]
        if dev_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for group in groups:
            subset = dev_data[dev_data["Group"] == group].sort_values("Voxels")
            if subset.empty:
                continue
            label = "Dense conv3d" if group == "DenseBaseline" else group
            ax.plot(
                subset["Voxels"],
                subset["Time (ms)"],
                marker="o",
                linewidth=2.5,
                markersize=8,
                label=label,
                color=color_map.get(group),
                linestyle=style_map.get(group, "-"),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(format_voxels))
        ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
        ax.set_xlabel("Grid Size (voxels)")
        ax.set_ylabel("Time")
        ax.set_title(f"Sparse Conv Scaling -- {device} (C=32, K=3x3x3)")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        out_name = f"conv_benchmark_throughput_{device.lower()}.png"
        print(f"Saving {out_name}")
        plt.savefig(out_name, bbox_inches="tight", dpi=150)
        plt.show()


def plot_sparsity_breakeven(df):
    """
    Plot 2: Sparse conv time vs occupancy, with dense conv3d as a horizontal reference.
    """
    sparse_df = df[df["Group"] == "Sparsity"].copy()
    dense_df = df[df["Group"] == "DenseRef"].copy()

    if sparse_df.empty:
        print("No data for sparsity breakeven plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = sns.color_palette("husl", 4)

    sparse_df = sparse_df.sort_values("Param")
    ax.plot(
        sparse_df["Param"],
        sparse_df["Time (ms)"],
        marker="o",
        linewidth=2.5,
        markersize=10,
        label="Sparse Conv",
        color=palette[0],
    )

    if not dense_df.empty:
        dense_time = dense_df["Time (ms)"].values[0]
        ax.axhline(
            y=dense_time,
            color=palette[2],
            linestyle="--",
            linewidth=2,
            label=f"Dense conv3d ({dense_time:.2f} ms)",
        )

        # Annotate crossover region
        above = sparse_df[sparse_df["Time (ms)"] >= dense_time]
        below = sparse_df[sparse_df["Time (ms)"] < dense_time]
        if not above.empty and not below.empty:
            cross_low = below["Param"].iloc[-1]
            cross_high = above["Param"].iloc[0]
            ax.axvspan(cross_low, cross_high, alpha=0.15, color="gray", label=f"Crossover ~{cross_low}-{cross_high}%")

    ax.set_xlabel("Occupancy (%)")
    ax.set_ylabel("Time")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.set_title("Sparsity Breakeven -- CUDA (bbox=32, C=32, K=3x3x3)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out_name = "conv_benchmark_sparsity_breakeven.png"
    print(f"Saving {out_name}")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


def plot_sparsity_breakeven_large(df):
    """
    Plot 2b: Large-scale sparsity breakeven -- one subplot per bbox size.
    This is where sparse convolution demonstrates its advantage over dense.
    """
    sparse_df = df[df["Group"] == "SparsityLarge"].copy()
    dense_df = df[df["Group"] == "DenseRefLarge"].copy()

    if sparse_df.empty:
        print("No data for large-scale sparsity breakeven plot.")
        return

    # Param = bbox_dim, Param2 = occupancy_pct for SparsityLarge
    # Param = bbox_dim for DenseRefLarge
    bbox_sizes = sorted(sparse_df["Param"].unique())

    sns.set_theme(style="whitegrid", context="talk")
    n_panels = len(bbox_sizes)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), squeeze=False)

    palette = sns.color_palette("husl", 4)

    for col, bbox_dim in enumerate(bbox_sizes):
        ax = axes[0][col]

        bbox_sparse = sparse_df[sparse_df["Param"] == bbox_dim].copy()
        bbox_sparse = bbox_sparse.sort_values("Param2")

        ax.plot(
            bbox_sparse["Param2"],
            bbox_sparse["Time (ms)"],
            marker="o",
            linewidth=2.5,
            markersize=10,
            label="Sparse Conv",
            color=palette[0],
        )

        bbox_dense = dense_df[dense_df["Param"] == bbox_dim]
        if not bbox_dense.empty:
            dense_time = bbox_dense["Time (ms)"].values[0]
            ax.axhline(
                y=dense_time,
                color=palette[2],
                linestyle="--",
                linewidth=2,
                label=f"Dense conv3d ({dense_time:.2f} ms)",
            )

            # Annotate crossover
            above = bbox_sparse[bbox_sparse["Time (ms)"] >= dense_time]
            below = bbox_sparse[bbox_sparse["Time (ms)"] < dense_time]
            if not below.empty and not above.empty:
                cross_low = below["Param2"].iloc[-1]
                cross_high = above["Param2"].iloc[0]
                ax.axvspan(
                    cross_low,
                    cross_high,
                    alpha=0.15,
                    color="gray",
                    label=f"Crossover ~{cross_low}-{cross_high}%",
                )
            elif not below.empty:
                # Sparse wins at all tested occupancies
                ax.annotate(
                    "Sparse wins at all\ntested occupancies",
                    xy=(below["Param2"].iloc[-1], below["Time (ms)"].iloc[-1]),
                    xytext=(0, -30),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="green"),
                )

        total_voxels = bbox_dim**3
        ax.set_xlabel("Occupancy (%)")
        ax.set_ylabel("Time")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
        ax.set_title(f"bbox={bbox_dim}^3 ({total_voxels:,} cells)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Sparsity Breakeven -- CUDA (C=32, K=3x3x3)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_name = "conv_benchmark_sparsity_large.png"
    print(f"Saving {out_name}")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


def plot_channel_scaling(df):
    """
    Plot 3: Time vs channel count on log-log axes.
    """
    chan_df = df[df["Group"] == "ChannelScale"].copy()
    if chan_df.empty:
        print("No data for channel scaling plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    devices = sorted(chan_df["Device"].unique())
    palette = sns.color_palette("husl", len(devices))
    color_map = dict(zip(devices, palette))

    for device in devices:
        subset = chan_df[chan_df["Device"] == device].sort_values("Param")
        if subset.empty:
            continue
        ax.plot(
            subset["Param"],
            subset["Time (ms)"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=device,
            color=color_map[device],
        )

    # Reference slopes
    x_ref = chan_df["Param"].unique()
    x_ref = np.sort(x_ref)
    if len(x_ref) >= 2:
        t0 = chan_df[chan_df["Param"] == x_ref[0]]["Time (ms)"].mean()
        y_linear = t0 * (x_ref / x_ref[0])
        y_quadratic = t0 * (x_ref / x_ref[0]) ** 2
        ax.plot(x_ref, y_linear, ":", alpha=0.4, color="gray", label="O(C) ref")
        ax.plot(x_ref, y_quadratic, "-.", alpha=0.4, color="gray", label="O(C^2) ref")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Channels (C)")
    ax.set_ylabel("Time")
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.set_title("Channel Scaling -- 16x16x16 grid, K=3x3x3")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out_name = "conv_benchmark_channel_scaling.png"
    print(f"Saving {out_name}")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


def plot_forward_vs_backward(df):
    """
    Plot 4: Forward vs backward time comparison at the largest grid size.
    """
    fwd_df = df[df["Group"] == "Forward"].copy()
    bwd_df = df[df["Group"] == "Backward"].copy()

    if fwd_df.empty or bwd_df.empty:
        print("No data for forward vs backward plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")

    devices = sorted(set(fwd_df["Device"].unique()) & set(bwd_df["Device"].unique()))
    if not devices:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_labels = []
    bar_times = []
    bar_colors = []
    palette = sns.color_palette("husl", 4)

    for i, device in enumerate(devices):
        fwd_dev = fwd_df[fwd_df["Device"] == device]
        bwd_dev = bwd_df[bwd_df["Device"] == device]
        if fwd_dev.empty or bwd_dev.empty:
            continue

        max_param = fwd_dev["Param"].max()
        fwd_time = fwd_dev[fwd_dev["Param"] == max_param]["Time (ms)"].values
        bwd_time = bwd_dev[bwd_dev["Param"] == max_param]["Time (ms)"].values

        if len(fwd_time) == 0 or len(bwd_time) == 0:
            continue

        fwd_t = fwd_time[0]
        bwd_t = bwd_time[0]
        ratio = bwd_t / fwd_t if fwd_t > 0 else 0

        bar_labels.extend([f"{device}\nForward", f"{device}\nBackward"])
        bar_times.extend([fwd_t, bwd_t])
        bar_colors.extend([palette[i * 2], palette[i * 2 + 1]])

        voxels = int(max_param**3)
        ax.annotate(
            f"ratio: {ratio:.2f}x",
            xy=(len(bar_labels) - 1.5, max(fwd_t, bwd_t)),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    if not bar_labels:
        return

    bars = ax.bar(bar_labels, bar_times, color=bar_colors)

    for bar, val in zip(bars, bar_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_times) * 0.02,
            f"{val:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Forward vs Backward -- C=32, K=3x3x3, grid={int(max_param)}^3")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_name = "conv_benchmark_fwd_vs_bwd.png"
    print(f"Saving {out_name}")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


# =============================================================================
# 5. Summary
# =============================================================================


def print_summary(df):
    """Print a text summary of benchmark results."""

    print("\n" + "=" * 72)
    print("  Sparse Convolution Benchmark Summary")
    print("=" * 72)

    # Grid-size sweep results
    grid_groups = ["Forward", "Backward", "DenseBaseline", "TopologyBuild"]
    grid_df = df[df["Group"].isin(grid_groups) & (df["Param"] > 0)]
    if not grid_df.empty:
        for device in sorted(grid_df["Device"].unique()):
            print(f"\n--- {device} ---")
            dev_data = grid_df[grid_df["Device"] == device]
            header = f"  {'Grid':>6s}  {'Voxels':>8s}"
            for g in grid_groups:
                label = "Dense" if g == "DenseBaseline" else g
                header += f"  {label:>12s}"
            print(header)

            for param in sorted(dev_data["Param"].unique()):
                voxels = int(param**3)
                row = f"  {int(param):>4d}^3  {voxels:>8d}"
                for g in grid_groups:
                    subset = dev_data[(dev_data["Group"] == g) & (dev_data["Param"] == param)]
                    if not subset.empty:
                        t = subset["Time (ms)"].values[0]
                        row += f"  {t:>10.3f}ms"
                    else:
                        row += f"  {'--':>12s}"
                print(row)

    # Sparsity results
    sparse_df = df[df["Group"] == "Sparsity"]
    dense_ref = df[df["Group"] == "DenseRef"]
    if not sparse_df.empty:
        print("\n--- Sparsity Breakeven (CUDA, bbox=32, C=32) ---")
        print(f"  {'Occupancy':>10s}  {'Voxels':>8s}  {'Time':>12s}")
        for _, row in sparse_df.sort_values("Param").iterrows():
            occ = int(row["Param"])
            vox = int(row.get("voxels", 0))
            t = row["Time (ms)"]
            print(f"  {occ:>9d}%  {vox:>8d}  {t:>10.3f}ms")
        if not dense_ref.empty:
            dt = dense_ref["Time (ms)"].values[0]
            print(f"  {'Dense ref':>10s}  {32**3:>8d}  {dt:>10.3f}ms")

    # Large-scale sparsity results
    sparse_large = df[df["Group"] == "SparsityLarge"]
    dense_large = df[df["Group"] == "DenseRefLarge"]
    if not sparse_large.empty:
        bbox_sizes = sorted(sparse_large["Param"].unique())
        for bbox_dim in bbox_sizes:
            total = bbox_dim**3
            print(f"\n--- Sparsity Breakeven (CUDA, bbox={bbox_dim}, {total:,} cells, C=32) ---")
            print(f"  {'Occupancy':>10s}  {'Voxels':>10s}  {'Sparse':>12s}  {'Dense':>12s}  {'Speedup':>8s}")
            bbox_sparse = sparse_large[sparse_large["Param"] == bbox_dim].sort_values("Param2")
            bbox_dense = dense_large[dense_large["Param"] == bbox_dim]
            dense_t = bbox_dense["Time (ms)"].values[0] if not bbox_dense.empty else float("nan")
            for _, row in bbox_sparse.iterrows():
                occ = int(row["Param2"])
                vox = int(row.get("voxels", 0))
                st = row["Time (ms)"]
                speedup = dense_t / st if st > 0 else 0
                marker = " <-- sparse wins" if speedup > 1 else ""
                print(f"  {occ:>9d}%  {vox:>10,d}  {st:>10.3f}ms  {dense_t:>10.3f}ms  {speedup:>7.2f}x{marker}")

    # Channel scaling
    chan_df = df[df["Group"] == "ChannelScale"]
    if not chan_df.empty:
        print("\n--- Channel Scaling (16^3 grid, K=3x3x3) ---")
        for device in sorted(chan_df["Device"].unique()):
            dev_data = chan_df[chan_df["Device"] == device].sort_values("Param")
            print(f"  {device}:")
            for _, row in dev_data.iterrows():
                c = int(row["Param"])
                t = row["Time (ms)"]
                print(f"    C={c:<4d}  {t:>10.3f}ms")

    print()


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GatherScatterDefault sparse convolution benchmark results.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=str, help="Path to the benchmark executable to run")
    group.add_argument("--file", type=str, help="Path to existing JSON results file")

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Regex to filter benchmark names. With --run, passed to Google Benchmark. "
        "With --file, filters results post-hoc.",
    )

    args = parser.parse_args()

    json_path = ""
    if args.run:
        json_path = run_benchmark_executable(args.run, benchmark_filter=args.filter)
    else:
        json_path = args.file

    df = load_benchmarks(json_path)

    if args.filter and not args.run:
        df = df[df["Name"].str.contains(args.filter, regex=True)]

    print(f"Loaded {len(df)} benchmark records.")

    if df.empty:
        print("No data found to plot.")
        sys.exit(0)

    print_summary(df)

    plot_throughput_vs_grid_size(df)
    plot_sparsity_breakeven(df)
    plot_sparsity_breakeven_large(df)
    plot_channel_scaling(df)
    plot_forward_vs_backward(df)
