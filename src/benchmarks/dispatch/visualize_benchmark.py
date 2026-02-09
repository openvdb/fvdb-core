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

# -----------------------------------------------------------------------------
# 1. Parsing Logic
# -----------------------------------------------------------------------------


def detect_benchmark_type(benchmarks):
    """Auto-detect benchmark type from naming patterns."""
    names = [b["name"] for b in benchmarks]

    # for_each benchmark detection
    if any("ForEach_" in n or "SoL_" in n for n in names):
        return "for_each"
    # Synthetic voxel benchmark detection
    if any("Uniform_" in n or "Unbalanced_" in n for n in names):
        return "synthetic"
    # CPU pool comparison detection
    if any("CPU_NoAlloc" in n for n in names):
        return "cpu_pool"
    # Fallback
    return "unknown"


def parse_for_each_benchmark_name(full_name):
    """
    Parse for_each benchmark names.
    Format: BM_ForEach_Softplus_<Contiguity>_<Device>_<Dtype>/<Size>
            BM_SoL_<Type>_<Device>_<Dtype>/<Size>
    Example: BM_ForEach_Softplus_Contiguous_CUDA_Float32/1000000
             BM_SoL_Memcpy_CPU_Float32/1000000
    """
    if full_name.startswith("BM_"):
        full_name = full_name[3:]

    parts = full_name.split("/")
    base_name = parts[0]
    size = int(parts[1]) if len(parts) > 1 else 0

    tokens = base_name.split("_")

    # Detect if this is a SoL baseline or a ForEach benchmark
    is_sol = tokens[0] == "SoL"

    device = "Unknown"
    if "CUDA" in tokens:
        device = "CUDA"
    elif "CPU" in tokens:
        device = "CPU"

    contiguity = "Contiguous"
    if "Strided" in tokens:
        contiguity = "Strided"

    dtype = "Float32"
    if "Float64" in tokens:
        dtype = "Float64"

    if is_sol:
        # SoL_<Type>_<Device>_<Dtype>
        impl = "SoL_" + tokens[1]
    else:
        # ForEach_Softplus_<Contiguity>_<Device>_<Dtype>
        impl = "ForEach"

    # Create a combined label for plotting
    label = f"{impl}_{contiguity}" if not is_sol else impl

    return {
        "Implementation": impl,
        "Contiguity": contiguity,
        "Device": device,
        "Dtype": dtype,
        "Label": label,
        "IsSoL": is_sol,
        "Size": size,
        "Name": base_name,
    }


def parse_synthetic_benchmark_name(full_name):
    """
    Parse synthetic voxel benchmark names.
    Format: BM_<Workload>_<Implementation>/<Size>
    """
    if full_name.startswith("BM_"):
        full_name = full_name[3:]

    parts = full_name.split("/")
    base_name = parts[0]
    size = int(parts[1]) if len(parts) > 1 else 0

    tokens = base_name.split("_")
    workload = tokens[0]
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
    """
    if full_name.startswith("BM_"):
        full_name = full_name[3:]

    parts = full_name.split("/")
    base_name = parts[0]
    size = int(parts[1]) if len(parts) > 1 else 0

    tokens = base_name.split("_")

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


def load_benchmarks(source, is_file=True):
    """Loads JSON data either from a file or directly from a string/dict."""
    try:
        if is_file:
            with open(source, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(source)

        benchmarks = data.get("benchmarks", [])

        bench_type = detect_benchmark_type(benchmarks)
        print(f"Detected benchmark type: {bench_type}")

        if bench_type == "for_each":
            parser = parse_for_each_benchmark_name
        elif bench_type == "synthetic":
            parser = parse_synthetic_benchmark_name
        elif bench_type == "cpu_pool":
            parser = parse_cpu_pool_benchmark_name
        else:
            parser = parse_for_each_benchmark_name

        parsed_records = []
        for b in benchmarks:
            if b.get("run_type") == "aggregate":
                continue

            info = parser(b["name"])

            throughput = b.get("items_per_second", 0)
            if throughput == 0 and info["Size"] > 0:
                time_unit = b.get("time_unit", "ns")
                time_val = b["real_time"]
                if time_unit == "ms":
                    time_val *= 1e6  # convert to ns
                elif time_unit == "us":
                    time_val *= 1e3
                throughput = info["Size"] / (time_val / 1e9)

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


# -----------------------------------------------------------------------------
# for_each benchmark plots
# -----------------------------------------------------------------------------


def plot_for_each_benchmarks(df):
    """Plot for_each + views benchmarks: contiguous vs strided, per device and dtype."""
    df = df[df["Throughput"] > 0].copy()
    sns.set_theme(style="whitegrid", context="talk")

    devices = df["Device"].unique()
    dtypes = df["Dtype"].unique()

    for device in devices:
        dev_data = df[df["Device"] == device]

        for dtype in dtypes:
            subset = dev_data[dev_data["Dtype"] == dtype].copy()
            if subset.empty:
                continue

            # Separate for_each results from SoL baselines
            fe_data = subset[~subset["IsSoL"]].copy() if "IsSoL" in subset.columns else subset
            sol_data = subset[subset["IsSoL"]].copy() if "IsSoL" in subset.columns else pd.DataFrame()

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # --- Plot 1: Throughput vs Size ---
            ax1 = axes[0]
            palette = sns.color_palette("husl", 4)
            color_idx = 0

            for contig in ["Contiguous", "Strided"]:
                contig_data = fe_data[fe_data["Contiguity"] == contig].sort_values("Size")
                if not contig_data.empty:
                    linestyle = "-" if contig == "Contiguous" else "--"
                    ax1.plot(
                        contig_data["Size"],
                        contig_data["Throughput"],
                        marker="o",
                        linewidth=2.5,
                        markersize=8,
                        label=f"for_each ({contig})",
                        linestyle=linestyle,
                        color=palette[color_idx],
                    )
                    color_idx += 1

            # Plot SoL baselines
            for _, sol_impl in enumerate(sol_data["Implementation"].unique()):
                impl_data = sol_data[sol_data["Implementation"] == sol_impl].sort_values("Size")
                ax1.plot(
                    impl_data["Size"],
                    impl_data["Throughput"],
                    marker="s",
                    linewidth=1.5,
                    markersize=6,
                    label=sol_impl.replace("SoL_", "SoL: "),
                    linestyle=":",
                    alpha=0.7,
                    color=palette[color_idx % len(palette)],
                )
                color_idx += 1

            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.xaxis.set_major_formatter(FuncFormatter(format_size))
            ax1.yaxis.set_major_formatter(FuncFormatter(format_billions))
            ax1.set_xlabel("Input Size (elements)")
            ax1.set_ylabel("Throughput (items/sec)")
            ax1.set_title(f"for_each Softplus — {device} {dtype}")
            ax1.legend(loc="upper left", fontsize=9)
            ax1.grid(True, which="both", alpha=0.3)

            # --- Plot 2: Contiguous vs Strided ratio at largest N ---
            ax2 = axes[1]
            largest_size = fe_data["Size"].max()
            bar_subset = fe_data[fe_data["Size"] == largest_size]

            contig_tp = bar_subset[bar_subset["Contiguity"] == "Contiguous"]["Throughput"].values
            strided_tp = bar_subset[bar_subset["Contiguity"] == "Strided"]["Throughput"].values

            labels = []
            values = []

            if len(contig_tp) > 0:
                labels.append("Contiguous")
                values.append(contig_tp[0])
            if len(strided_tp) > 0:
                labels.append("Strided")
                values.append(strided_tp[0])

            # Add SoL baselines to the bar chart
            for sol_impl in sol_data["Implementation"].unique():
                sol_at_size = sol_data[
                    (sol_data["Implementation"] == sol_impl) & (sol_data["Size"] == largest_size)
                ]
                if not sol_at_size.empty:
                    labels.append(sol_impl.replace("SoL_", "SoL: "))
                    values.append(sol_at_size["Throughput"].values[0])

            if values:
                colors = sns.color_palette("husl", len(labels))
                bars = ax2.barh(labels, values, color=colors)

                max_val = max(values) if values else 1
                for bar, val in zip(bars, values):
                    tp_str = (
                        f"{val / 1e9:.2f} G/s"
                        if val >= 1e9
                        else f"{val / 1e6:.0f} M/s"
                    )
                    ax2.text(
                        bar.get_width() + max_val * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        tp_str,
                        va="center",
                        fontsize=10,
                    )

                ax2.set_xlabel(f"Throughput at N={largest_size / 1e6:.0f}M")
                ax2.set_title(f"Contiguous vs Strided — {device} {dtype}")

            plt.tight_layout()
            out_name = f"benchmark_for_each_{device.lower()}_{dtype.lower()}.png"
            print(f"Saving plot to {out_name}...")
            plt.savefig(out_name, bbox_inches="tight", dpi=150)
            plt.show()

    # Combined device comparison
    plot_for_each_device_comparison(df)


def plot_for_each_device_comparison(df):
    """Combined bar chart comparing CPU vs CUDA for contiguous float32 at largest N."""
    df = df[df["Throughput"] > 0].copy()
    fe_data = df[~df.get("IsSoL", False)].copy() if "IsSoL" in df.columns else df

    # Filter to float32 contiguous only
    subset = fe_data[(fe_data["Dtype"] == "Float32") & (fe_data["Contiguity"] == "Contiguous")]
    if subset.empty:
        return

    largest_size = subset["Size"].max()
    comparison = subset[subset["Size"] == largest_size]

    if comparison.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    devices = comparison["Device"].unique()
    throughputs = [
        comparison[comparison["Device"] == d]["Throughput"].values[0] / 1e9
        for d in devices
        if len(comparison[comparison["Device"] == d]) > 0
    ]

    colors = sns.color_palette("husl", len(devices))
    bars = ax.barh(list(devices), throughputs, color=colors)

    for bar, val in zip(bars, throughputs):
        ax.text(
            bar.get_width() + max(throughputs) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} G/s",
            va="center",
            fontsize=11,
        )

    ax.set_xlabel("Throughput (G items/sec)")
    ax.set_title(f"for_each Softplus — CPU vs CUDA (Float32, Contiguous, N={largest_size / 1e6:.0f}M)")

    plt.tight_layout()
    out_name = "benchmark_for_each_device_comparison.png"
    print(f"Saving plot to {out_name}...")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# Synthetic benchmark plots
# -----------------------------------------------------------------------------


def plot_synthetic_benchmarks(df):
    """Specialized plot for synthetic voxel benchmarks."""
    df = df[df["Throughput"] > 0].copy()
    sns.set_theme(style="whitegrid", context="talk")

    workloads = df["Workload"].unique()

    impls = df["Implementation"].unique()
    palette = sns.color_palette("husl", len(impls))
    color_map = dict(zip(impls, palette))

    for workload in workloads:
        subset = df[df["Workload"] == workload].copy()
        if subset.empty:
            continue

        serial_data = subset[subset["Implementation"] == "Serial"]
        if not serial_data.empty:
            serial_throughput = serial_data.set_index("Size")["Throughput"].to_dict()
            subset["Speedup"] = subset.apply(
                lambda row: row["Throughput"] / serial_throughput.get(row["Size"], 1), axis=1
            )
        else:
            subset["Speedup"] = 1

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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

        ax2 = axes[1]
        largest_size = subset["Size"].max()
        bar_data = subset[subset["Size"] == largest_size].copy()
        bar_data = bar_data.sort_values("Speedup", ascending=True)

        colors = [color_map.get(impl) for impl in bar_data["Implementation"]]
        bars = ax2.barh(bar_data["Implementation"], bar_data["Speedup"], color=colors)

        for bar, speedup in zip(bars, bar_data["Speedup"]):
            ax2.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{speedup:.1f}x", va="center", fontsize=10
            )

        ax2.set_xlabel(f"Speedup vs Serial (N={largest_size / 1e6:.0f}M)")
        ax2.set_title(f"{workload} Workload - Parallel Speedup")
        ax2.axvline(x=1, color="gray", linestyle="--", alpha=0.5)

        max_speedup = bar_data["Speedup"].max()
        ax2.set_xlim(0, max(max_speedup * 1.15, 32))

        plt.tight_layout()
        out_name = f"benchmark_synthetic_{workload.lower()}.png"
        print(f"Saving plot to {out_name}...")
        plt.savefig(out_name, bbox_inches="tight", dpi=150)
        plt.show()

    plot_synthetic_comparison(df, color_map)


def plot_synthetic_comparison(df, color_map):
    """Create a combined comparison of all implementations across workloads."""
    df = df[df["Throughput"] > 0].copy()

    largest_size = df["Size"].max()
    comparison = df[df["Size"] == largest_size].copy()

    if comparison.empty:
        return

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
    ax.set_title(f"Implementation Comparison at N={largest_size / 1e6:.0f}M")
    ax.set_xticks(x)
    ax.set_xticklabels(impls, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_name = "benchmark_synthetic_comparison.png"
    print(f"Saving plot to {out_name}...")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# CPU pool benchmark plots
# -----------------------------------------------------------------------------


def plot_cpu_pool_benchmarks(df):
    """Specialized plot for CPU pool comparison benchmarks."""
    df = df[df["Throughput"] > 0].copy()
    sns.set_theme(style="whitegrid", context="talk")

    impls = df["Implementation"].unique()
    palette = sns.color_palette("husl", len(impls))
    color_map = dict(zip(impls, palette))

    serial_data = df[df["Implementation"] == "Serial"]
    if not serial_data.empty:
        serial_throughput = serial_data.set_index("Size")["Throughput"].to_dict()
        df["Speedup"] = df.apply(
            lambda row: row["Throughput"] / serial_throughput.get(row["Size"], 1), axis=1
        )
    else:
        df["Speedup"] = 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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

    ax2 = axes[1]
    largest_size = df["Size"].max()
    bar_data = df[df["Size"] == largest_size].copy()
    bar_data = bar_data.sort_values("Speedup", ascending=True)

    colors = [color_map.get(impl) for impl in bar_data["Implementation"]]
    bars = ax2.barh(bar_data["Implementation"], bar_data["Speedup"], color=colors)

    for bar, speedup in zip(bars, bar_data["Speedup"]):
        ax2.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{speedup:.1f}x", va="center", fontsize=10
        )

    ax2.set_xlabel(f"Speedup vs Serial (N={largest_size / 1e6:.0f}M)")
    ax2.set_title("Parallel Speedup")
    ax2.axvline(x=1, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_name = "benchmark_cpu_pool.png"
    print(f"Saving plot to {out_name}...")
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------


def plot_benchmarks(df, title_suffix=""):
    """Route to appropriate plotting function based on benchmark type."""
    bench_type = df["bench_type"].iloc[0] if "bench_type" in df.columns else "unknown"

    if bench_type == "for_each":
        plot_for_each_benchmarks(df)
    elif bench_type == "synthetic":
        plot_synthetic_benchmarks(df)
    elif bench_type == "cpu_pool":
        plot_cpu_pool_benchmarks(df)
    else:
        print(f"Unknown benchmark type: {bench_type}")


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

    if args.filter and not args.run:
        df = df[df["Name"].str.contains(args.filter, regex=True)]

    print(f"Loaded {len(df)} benchmark records.")

    if df.empty:
        print("No data found to plot.")
    else:
        bench_type = df["bench_type"].iloc[0] if "bench_type" in df.columns else "unknown"

        # --- Summary ---
        if bench_type == "for_each":
            print("\n--- for_each Benchmark Summary ---")
            for device in df["Device"].unique():
                dev_data = df[df["Device"] == device]
                for dtype in dev_data["Dtype"].unique():
                    dtype_data = dev_data[dev_data["Dtype"] == dtype]
                    largest_n = dtype_data["Size"].max()
                    print(f"\n{device} {dtype} (N={largest_n / 1e6:.0f}M):")

                    at_largest = dtype_data[dtype_data["Size"] == largest_n].copy()
                    at_largest = at_largest.sort_values("Throughput", ascending=False)

                    for _, row in at_largest.iterrows():
                        tp_str = (
                            f"{row['Throughput'] / 1e9:.2f} G/s"
                            if row["Throughput"] >= 1e9
                            else f"{row['Throughput'] / 1e6:.0f} M/s"
                        )
                        is_sol = row.get("IsSoL", False)
                        prefix = "[SoL] " if is_sol else "      "
                        label = row.get("Label", row["Name"])
                        print(f"  {prefix}{label:35s} {tp_str:>12s}")

        elif bench_type == "synthetic":
            print("\n--- Synthetic Benchmark Summary ---")
            for workload in df["Workload"].unique():
                wl_data = df[df["Workload"] == workload]
                largest_n = wl_data["Size"].max()
                print(f"\n{workload} Workload (N={largest_n / 1e6:.0f}M):")

                summary = wl_data[wl_data["Size"] == largest_n][["Implementation", "Throughput"]].copy()
                summary = summary.sort_values("Throughput", ascending=False)
                summary["Throughput (M/s)"] = summary["Throughput"] / 1e6

                serial_tp = summary[summary["Implementation"] == "Serial"]["Throughput"].values
                if len(serial_tp) > 0:
                    summary["Speedup"] = summary["Throughput"] / serial_tp[0]
                else:
                    summary["Speedup"] = 1

                for _, row in summary.iterrows():
                    print(
                        f"  {row['Implementation']:25s} "
                        f"{row['Throughput (M/s)']:8.1f} M/s  "
                        f"({row['Speedup']:5.1f}x)"
                    )

        elif bench_type == "cpu_pool":
            print("\n--- CPU Pool Comparison Summary ---")
            largest_n = df["Size"].max()
            print(f"At N={largest_n / 1e6:.0f}M:")

            summary = df[df["Size"] == largest_n][["Implementation", "Throughput"]].copy()
            summary = summary.sort_values("Throughput", ascending=False)

            serial_tp = summary[summary["Implementation"] == "Serial"]["Throughput"].values
            if len(serial_tp) > 0:
                summary["Speedup"] = summary["Throughput"] / serial_tp[0]
            else:
                summary["Speedup"] = 1

            for _, row in summary.iterrows():
                tp_str = (
                    f"{row['Throughput'] / 1e9:.2f} G/s"
                    if row["Throughput"] >= 1e9
                    else f"{row['Throughput'] / 1e6:.0f} M/s"
                )
                print(f"  {row['Implementation']:25s} {tp_str:>12s}  ({row['Speedup']:5.1f}x)")

        print("\n")
        plot_benchmarks(df)
