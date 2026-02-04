import argparse
import json
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# -----------------------------------------------------------------------------
# 1. Parsing Logic
# -----------------------------------------------------------------------------


def parse_benchmark_name(full_name):
    """
    Deconstructs the Google Benchmark name string into meaningful columns.
    Expected Format: BM_<Impl>_<InputType>_<Device>[_Variant]/<Size>
    Example: BM_TorchGelu_Contiguous_CUDA_NoAlloc/1024
    """
    # Remove "BM_" prefix
    if full_name.startswith("BM_"):
        full_name = full_name[3:]

    # Split Size (Google Benchmark uses /size for the Arg argument)
    parts = full_name.split("/")
    base_name = parts[0]
    size = int(parts[1]) if len(parts) > 1 else 0

    # Split the underscores.
    # Assumption: The naming convention in C++ was consistent.
    # We expect: Impl_Input_Device[_Variant]
    tokens = base_name.split("_")

    # Heuristic parsing based on your specific naming scheme
    impl = tokens[0]  # e.g., TorchGelu, GeluNew, GeluOld

    # Find Device (CPU/CUDA)
    device = "Unknown"
    if "CUDA" in tokens:
        device = "CUDA"
    elif "CPU" in tokens:
        device = "CPU"

    # Find Variant (NoAlloc)
    variant = "Alloc"
    if "NoAlloc" in tokens:
        variant = "NoAlloc"

    # Find Input Type (Contiguous, Strided, Float16)
    input_type = "Contiguous"  # Default
    if "Strided" in tokens:
        input_type = "Strided"
    elif "Float16" in tokens:
        input_type = "Float16"

    return {
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

        parsed_records = []
        for b in benchmarks:
            # Skip aggregates (mean, median) if they exist
            if b.get("run_type") == "aggregate":
                continue

            info = parse_benchmark_name(b["name"])

            # Google Benchmark JSON usually provides these fields
            # items_per_second is optional in C++, but you have it enabled
            throughput = b.get("items_per_second", 0)

            # If items_per_second isn't there, calculate it if possible
            if throughput == 0 and info["Size"] > 0:
                # real_time is in nanoseconds
                throughput = info["Size"] / (b["real_time"] / 1e9)

            info["Throughput"] = throughput
            info["Real Time (ns)"] = b["real_time"]
            parsed_records.append(info)

        return pd.DataFrame(parsed_records)

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
    return f"{x * 1e-9:.0f}G"


def plot_benchmarks(df, title_suffix=""):
    # Filter out 0 throughput (sanity check)
    df = df[df["Throughput"] > 0]

    # Setup styles
    sns.set_theme(style="whitegrid", context="talk")

    # We generally want to separate "With Allocation" and "No Allocation"
    # because the scales are vastly different.
    alloc_modes = df["Allocation"].unique()

    for mode in alloc_modes:
        subset = df[df["Allocation"] == mode]
        if subset.empty:
            continue

        # Create a FacetGrid or Subplots based on Device/Input Type
        # We'll map "Input Type" to columns
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

        # Adjust axes
        g.set(xscale="log")

        # Formatting
        for ax in g.axes.flat:
            ax.yaxis.set_major_formatter(FuncFormatter(format_billions))
            ax.grid(True, which="minor", linestyle="--", alpha=0.3)

        g.fig.suptitle(f"GeLU Throughput - {mode} Mode {title_suffix}", y=1.02, fontsize=20, weight="bold")
        g.set_axis_labels("Input Tensor Size (elements)", "Throughput (Items/sec)")

        # Save or Show
        out_name = f"benchmark_plot_{mode}.png"
        print(f"Saving plot to {out_name}...")
        plt.savefig(out_name, bbox_inches="tight", dpi=150)
        plt.show()


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
        # Print a quick summary stats table to console
        print("\n--- Summary (Max Throughput per Impl/Device) ---")
        summary = df.groupby(["Implementation", "Device", "Allocation"])["Throughput"].max() / 1e9
        print(summary.to_string(float_format="{:.2f} G/s".format))
        print("\n")

        plot_benchmarks(df)
