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
from matplotlib.ticker import FuncFormatter

# -----------------------------------------------------------------------------
# 1. Parsing Logic
# -----------------------------------------------------------------------------


def detect_benchmark_type(benchmarks):
    """Auto-detect benchmark type from naming patterns."""
    names = [b["name"] for b in benchmarks]

    # Synthetic voxel benchmark detection
    if any("Uniform_" in n or "Unbalanced_" in n for n in names):
        return "synthetic"
    # CPU pool comparison detection
    if any("CPU_NoAlloc" in n for n in names):
        return "cpu_pool"
    # Default GELU benchmark
    return "gelu"


def parse_synthetic_benchmark_name(full_name):
    """
    Parse synthetic voxel benchmark names.
    Format: BM_<Workload>_<Implementation>/<Size>
    Example: BM_Uniform_OpenMP/10000000, BM_Unbalanced_WorkStealing/1000000
    """
    if full_name.startswith("BM_"):
        full_name = full_name[3:]

    parts = full_name.split("/")
    base_name = parts[0]
    size = int(parts[1]) if len(parts) > 1 else 0

    tokens = base_name.split("_")

    # First token is workload type (Uniform/Unbalanced)
    workload = tokens[0]

    # Rest is implementation name
    impl = "_".join(tokens[1:])

    return {
        "Workload": workload,
        "Implementation": impl,
        "Device": "CPU",
        "Input Type": "Contiguous",
        "Allocation": "NoAlloc",
        "Size": size,
        "Name": base_name,
    }


def parse_cpu_pool_benchmark_name(full_name):
    """
    Parse CPU pool comparison benchmark names.
    Format: BM_<Implementation>_CPU_NoAlloc/<Size>
    Example: BM_Torch_CPU_NoAlloc/10000000
    """
    if full_name.startswith("BM_"):
        full_name = full_name[3:]

    parts = full_name.split("/")
    base_name = parts[0]
    size = int(parts[1]) if len(parts) > 1 else 0

    tokens = base_name.split("_")

    # Find CPU position and extract implementation
    try:
        cpu_idx = tokens.index("CPU")
        impl = "_".join(tokens[:cpu_idx])
    except ValueError:
        impl = tokens[0]

    return {
        "Workload": "GELU",
        "Implementation": impl,
        "Device": "CPU",
        "Input Type": "Contiguous",
        "Allocation": "NoAlloc",
        "Size": size,
        "Name": base_name,
    }


def parse_gelu_benchmark_name(full_name):
    """
    Parse original GELU benchmark names.
    Format: BM_<Impl>_<InputType>_<Device>[_Variant]/<Size>
    Example: BM_TorchGelu_Contiguous_CUDA_NoAlloc/1024
    """
    if full_name.startswith("BM_"):
        full_name = full_name[3:]

    parts = full_name.split("/")
    base_name = parts[0]
    size = int(parts[1]) if len(parts) > 1 else 0

    tokens = base_name.split("_")
    impl = tokens[0]

    device = "Unknown"
    if "CUDA" in tokens:
        device = "CUDA"
    elif "CPU" in tokens:
        device = "CPU"

    variant = "Alloc"
    if "NoAlloc" in tokens:
        variant = "NoAlloc"

    input_type = "Contiguous"
    if "Strided" in tokens:
        input_type = "Strided"
    elif "Float16" in tokens:
        input_type = "Float16"

    return {
        "Workload": "GELU",
        "Implementation": impl,
        "Device": device,
        "Input Type": input_type,
        "Allocation": variant,
        "Size": size,
        "Name": base_name,
    }


def load_benchmarks(source, is_file=True):
    """Loads JSON data either from a file or directly from a string/dict."""
    try:
        if is_file:
            with open(source, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(source)

        benchmarks = data.get("benchmarks", [])

        # Auto-detect benchmark type
        bench_type = detect_benchmark_type(benchmarks)
        print(f"Detected benchmark type: {bench_type}")

        # Select appropriate parser
        if bench_type == "synthetic":
            parser = parse_synthetic_benchmark_name
        elif bench_type == "cpu_pool":
            parser = parse_cpu_pool_benchmark_name
        else:
            parser = parse_gelu_benchmark_name

        parsed_records = []
        for b in benchmarks:
            # Skip aggregates (mean, median) if they exist
            if b.get("run_type") == "aggregate":
                continue

            info = parser(b["name"])

            # Google Benchmark JSON usually provides these fields
            throughput = b.get("items_per_second", 0)

            # If items_per_second isn't there, calculate it if possible
            if throughput == 0 and info["Size"] > 0:
                throughput = info["Size"] / (b["real_time"] / 1e9)

            info["Throughput"] = throughput
            info["Real Time (ns)"] = b["real_time"]
            info["Time (ms)"] = b["real_time"] / 1e6
            parsed_records.append(info)

        df = pd.DataFrame(parsed_records)
        df["bench_type"] = bench_type
        return df

    except FileNotFoundError:
        print(f"Error: File {source} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {source}.")
        sys.exit(1)


# -----------------------------------------------------------------------------
# 2. Execution Logic
# -----------------------------------------------------------------------------


def run_benchmark_executable(executable_path, output_file="results.json", benchmark_filter=None):
    """Runs the C++ benchmark binary and saves JSON output."""
    if not os.path.exists(executable_path) and not executable_path.startswith("./"):
        # Try adding ./ if strictly local
        if os.path.exists("./" + executable_path):
            executable_path = "./" + executable_path

    print(f"Running benchmark: {executable_path}")
    print(f"Outputting to: {output_file}")

    # Google Benchmark flag to export JSON
    cmd = [executable_path, f"--benchmark_out={output_file}", "--benchmark_out_format=json"]

    # Pass filter to Google Benchmark executable to only run matching benchmarks
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


# -----------------------------------------------------------------------------
# 3. Visualization Logic
# -----------------------------------------------------------------------------


def format_billions(x, pos):
    """Formatter for Y-axis to show 10G, 20G, etc."""
    if x >= 1e9:
        return f"{x * 1e-9:.1f}G"
    elif x >= 1e6:
        return f"{x * 1e-6:.0f}M"
    else:
        return f"{x:.0f}"


def format_size(x, pos):
    """Formatter for X-axis to show 100K, 1M, 10M."""
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    elif x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def plot_synthetic_benchmarks(df):
    """Specialized plot for synthetic voxel benchmarks."""
    df = df[df["Throughput"] > 0].copy()
    sns.set_theme(style="whitegrid", context="talk")

    workloads = df["Workload"].unique()

    # Define a consistent color palette for implementations
    impls = df["Implementation"].unique()
    palette = sns.color_palette("husl", len(impls))
    color_map = dict(zip(impls, palette))

    for workload in workloads:
        subset = df[df["Workload"] == workload].copy()
        if subset.empty:
            continue

        # Calculate speedup relative to Serial
        serial_data = subset[subset["Implementation"] == "Serial"]
        if not serial_data.empty:
            serial_throughput = serial_data.set_index("Size")["Throughput"].to_dict()
            subset["Speedup"] = subset.apply(
                lambda row: row["Throughput"] / serial_throughput.get(row["Size"], 1), axis=1
            )
        else:
            subset["Speedup"] = 1

        # Create figure with two subplots: Throughput and Speedup
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Throughput vs Size (log-log)
        ax1 = axes[0]
        for impl in subset["Implementation"].unique():
            impl_data = subset[subset["Implementation"] == impl].sort_values("Size")
            ax1.plot(
                impl_data["Size"],
                impl_data["Throughput"],
                marker="o",
                linewidth=2.5,
                markersize=8,
                label=impl,
                color=color_map.get(impl),
            )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.xaxis.set_major_formatter(FuncFormatter(format_size))
        ax1.yaxis.set_major_formatter(FuncFormatter(format_billions))
        ax1.set_xlabel("Input Size (elements)")
        ax1.set_ylabel("Throughput (items/sec)")
        ax1.set_title(f"{workload} Workload - Throughput")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, which="both", alpha=0.3)

        # Plot 2: Speedup vs Size (bar chart at largest N)
        ax2 = axes[1]
        largest_size = subset["Size"].max()
        bar_data = subset[subset["Size"] == largest_size].copy()
        bar_data = bar_data.sort_values("Speedup", ascending=True)

        colors = [color_map.get(impl) for impl in bar_data["Implementation"]]
        bars = ax2.barh(bar_data["Implementation"], bar_data["Speedup"], color=colors)

        # Add value labels on bars
        for bar, speedup in zip(bars, bar_data["Speedup"]):
            ax2.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{speedup:.1f}x", va="center", fontsize=10
            )

        ax2.set_xlabel(f"Speedup vs Serial (N={largest_size/1e6:.0f}M)")
        ax2.set_title(f"{workload} Workload - Parallel Speedup")
        ax2.axvline(x=1, color="gray", linestyle="--", alpha=0.5)

        # Add ideal scaling line (32 threads)
        max_speedup = bar_data["Speedup"].max()
        ax2.set_xlim(0, max(max_speedup * 1.15, 32))

        plt.tight_layout()
        out_name = f"benchmark_synthetic_{workload.lower()}.png"
        print(f"Saving plot to {out_name}...")
        plt.savefig(out_name, bbox_inches="tight", dpi=150)
        plt.show()

    # Create combined comparison chart
    plot_synthetic_comparison(df, color_map)


def plot_synthetic_comparison(df, color_map):
    """Create a combined comparison of all implementations across workloads."""
    df = df[df["Throughput"] > 0].copy()

    largest_size = df["Size"].max()
    comparison = df[df["Size"] == largest_size].copy()

    if comparison.empty:
        return

    # Pivot for grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    workloads = comparison["Workload"].unique()
    impls = comparison["Implementation"].unique()

    x = np.arange(len(impls))
    width = 0.35

    for i, workload in enumerate(workloads):
        wl_data = comparison[comparison["Workload"] == workload]
        throughputs = []
        for impl in impls:
            impl_data = wl_data[wl_data["Implementation"] == impl]
            throughputs.append(impl_data["Throughput"].values[0] / 1e6 if len(impl_data) > 0 else 0)

        offset = (i - len(workloads) / 2 + 0.5) * width
        bars = ax.bar(x + offset, throughputs, width, label=workload)

        # Add value labels
        for bar, val in zip(bars, throughputs):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Throughput (M items/sec)")
    ax.set_xlabel("Implementation")
    ax.set_title(f"Implementation Comparison at N={largest_size/1e6:.0f}M")
    ax.set_xticks(x)
    ax.set_xticklabels(impls, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_name = "benchmark_synthetic_comparison.png"
    print(f"Saving plot to {out_name}...")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


def plot_cpu_pool_benchmarks(df):
    """Specialized plot for CPU pool comparison benchmarks."""
    df = df[df["Throughput"] > 0].copy()
    sns.set_theme(style="whitegrid", context="talk")

    impls = df["Implementation"].unique()
    palette = sns.color_palette("husl", len(impls))
    color_map = dict(zip(impls, palette))

    # Calculate speedup relative to Serial
    serial_data = df[df["Implementation"] == "Serial"]
    if not serial_data.empty:
        serial_throughput = serial_data.set_index("Size")["Throughput"].to_dict()
        df["Speedup"] = df.apply(lambda row: row["Throughput"] / serial_throughput.get(row["Size"], 1), axis=1)
    else:
        df["Speedup"] = 1

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Throughput vs Size
    ax1 = axes[0]
    for impl in df["Implementation"].unique():
        impl_data = df[df["Implementation"] == impl].sort_values("Size")
        ax1.plot(
            impl_data["Size"],
            impl_data["Throughput"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=impl,
            color=color_map.get(impl),
        )

    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(FuncFormatter(format_size))
    ax1.yaxis.set_major_formatter(FuncFormatter(format_billions))
    ax1.set_xlabel("Input Size (elements)")
    ax1.set_ylabel("Throughput (items/sec)")
    ax1.set_title("GELU Throughput by Implementation")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # Plot 2: Speedup bar chart at largest N
    ax2 = axes[1]
    largest_size = df["Size"].max()
    bar_data = df[df["Size"] == largest_size].copy()
    bar_data = bar_data.sort_values("Speedup", ascending=True)

    colors = [color_map.get(impl) for impl in bar_data["Implementation"]]
    bars = ax2.barh(bar_data["Implementation"], bar_data["Speedup"], color=colors)

    for bar, speedup in zip(bars, bar_data["Speedup"]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{speedup:.1f}x", va="center", fontsize=10)

    ax2.set_xlabel(f"Speedup vs Serial (N={largest_size/1e6:.0f}M)")
    ax2.set_title("Parallel Speedup")
    ax2.axvline(x=1, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_name = "benchmark_cpu_pool.png"
    print(f"Saving plot to {out_name}...")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


def plot_gelu_benchmarks(df, title_suffix=""):
    """Original plot for GELU benchmarks."""
    df = df[df["Throughput"] > 0]
    sns.set_theme(style="whitegrid", context="talk")

    alloc_modes = df["Allocation"].unique()

    for mode in alloc_modes:
        subset = df[df["Allocation"] == mode]
        if subset.empty:
            continue

        g = sns.relplot(
            data=subset,
            x="Size",
            y="Throughput",
            hue="Implementation",
            style="Implementation",
            col="Device",
            row="Input Type",
            kind="line",
            markers=True,
            dashes=False,
            height=5,
            aspect=1.4,
            linewidth=3,
        )

        g.set(xscale="log")

        for ax in g.axes.flat:
            ax.yaxis.set_major_formatter(FuncFormatter(format_billions))
            ax.grid(True, which="minor", linestyle="--", alpha=0.3)

        g.fig.suptitle(f"GeLU Throughput - {mode} Mode {title_suffix}", y=1.02, fontsize=20, weight="bold")
        g.set_axis_labels("Input Tensor Size (elements)", "Throughput (Items/sec)")

        out_name = f"benchmark_plot_{mode}.png"
        print(f"Saving plot to {out_name}...")
        plt.savefig(out_name, bbox_inches="tight", dpi=150)
        plt.show()


def plot_benchmarks(df, title_suffix=""):
    """Route to appropriate plotting function based on benchmark type."""
    bench_type = df["bench_type"].iloc[0] if "bench_type" in df.columns else "gelu"

    if bench_type == "synthetic":
        plot_synthetic_benchmarks(df)
    elif bench_type == "cpu_pool":
        plot_cpu_pool_benchmarks(df)
    else:
        plot_gelu_benchmarks(df, title_suffix)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Google Benchmark JSON results.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=str, help="Path to the benchmark executable to run (e.g., ./bench)")
    group.add_argument("--file", type=str, help="Path to existing JSON results file")

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Regex to filter benchmark names. With --run, passed to Google Benchmark "
        "to only execute matching benchmarks. With --file, filters results post-hoc.",
    )

    args = parser.parse_args()

    json_path = ""

    if args.run:
        json_path = run_benchmark_executable(args.run, benchmark_filter=args.filter)
    else:
        json_path = args.file

    df = load_benchmarks(json_path)

    # Optional post-hoc filtering (useful when loading from file, or for additional filtering)
    if args.filter and not args.run:
        df = df[df["Name"].str.contains(args.filter, regex=True)]

    print(f"Loaded {len(df)} benchmark records.")

    if df.empty:
        print("No data found to plot.")
    else:
        bench_type = df["bench_type"].iloc[0] if "bench_type" in df.columns else "gelu"

        if bench_type == "synthetic":
            print("\n--- Synthetic Benchmark Summary ---")
            for workload in df["Workload"].unique():
                wl_data = df[df["Workload"] == workload]
                largest_n = wl_data["Size"].max()
                print(f"\n{workload} Workload (N={largest_n/1e6:.0f}M):")

                summary = wl_data[wl_data["Size"] == largest_n][["Implementation", "Throughput"]].copy()
                summary = summary.sort_values("Throughput", ascending=False)
                summary["Throughput (M/s)"] = summary["Throughput"] / 1e6

                # Calculate speedup vs serial
                serial_tp = summary[summary["Implementation"] == "Serial"]["Throughput"].values
                if len(serial_tp) > 0:
                    summary["Speedup"] = summary["Throughput"] / serial_tp[0]
                else:
                    summary["Speedup"] = 1

                for _, row in summary.iterrows():
                    print(f"  {row['Implementation']:25s} {row['Throughput (M/s)']:8.1f} M/s  ({row['Speedup']:5.1f}x)")

        elif bench_type == "cpu_pool":
            print("\n--- CPU Pool Comparison Summary ---")
            largest_n = df["Size"].max()
            print(f"At N={largest_n/1e6:.0f}M:")

            summary = df[df["Size"] == largest_n][["Implementation", "Throughput"]].copy()
            summary = summary.sort_values("Throughput", ascending=False)

            serial_tp = summary[summary["Implementation"] == "Serial"]["Throughput"].values
            if len(serial_tp) > 0:
                summary["Speedup"] = summary["Throughput"] / serial_tp[0]
            else:
                summary["Speedup"] = 1

            for _, row in summary.iterrows():
                tp_str = (
                    f"{row['Throughput']/1e9:.2f} G/s"
                    if row["Throughput"] >= 1e9
                    else f"{row['Throughput']/1e6:.0f} M/s"
                )
                print(f"  {row['Implementation']:25s} {tp_str:>12s}  ({row['Speedup']:5.1f}x)")

        else:
            print("\n--- Summary (Max Throughput per Impl/Device) ---")
            summary = df.groupby(["Implementation", "Device", "Allocation"])["Throughput"].max() / 1e9
            print(summary.to_string(float_format="{:.2f} G/s".format))

        print("\n")
        plot_benchmarks(df)
