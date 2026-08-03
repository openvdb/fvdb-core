# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Measure sparse convolution topology construction for issue #668.

Run this from an installed fVDB environment, for example::

    python src/benchmarks/convolution/measure_topology_resources.py \
        --output /tmp/convolution_topology_resources.json

The driver starts a fresh subprocess for every CPU sample so ``ru_maxrss`` has
an idle, per-case high-water baseline. CUDA samples use PyTorch's allocator
peak counters above an idle baseline. The script deliberately bounds generated
transpose output and refuses cases whose exact output estimate exceeds the
configured cap; it is a release measurement, not a stress allocator.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


_CASES = (
    ("ks2_direct", (2, 2, 2), (2, 2, 2), "forward"),
    ("ks3_direct", (3, 3, 3), (3, 3, 3), "forward"),
    ("ks4_direct", (4, 4, 4), (4, 4, 4), "forward"),
    ("issue_668", (4, 1, 1), (4, 1, 1), "forward"),
    ("k_lt_s_count_fill", (3, 3, 3), (4, 4, 4), "forward"),
    ("k_gt_s_count_fill", (4, 4, 4), (3, 3, 3), "forward"),
    ("generated_transpose", (4, 4, 4), (4, 4, 4), "transpose"),
)


@dataclass(frozen=True)
class Sample:
    case: str
    kind: str
    device: str
    side: int
    input_voxels: int
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    candidate_rows: int
    output_voxels: int
    wall_time_ms: float
    cpu_peak_delta_bytes: int | None
    cuda_peak_allocated_delta_bytes: int | None
    cuda_peak_reserved_delta_bytes: int | None
    final_grid_bytes: int | None
    resource_stats: dict[str, Any] | None
    execution_median_ms: float | None


def _dense_ijk(side: int, device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, dtype=torch.int32, device=device)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)


def _grid_total_bytes(grid: Any) -> int | None:
    for name in ("total_bytes", "nbytes"):
        value = getattr(grid, name, None)
        if value is not None:
            return int(value() if callable(value) else value)
    return None


def _hwm_bytes() -> int:
    # Linux reports ru_maxrss in KiB; fVDB's CI/release workers are Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_worker(args: argparse.Namespace) -> Sample:
    import fvdb
    from fvdb import _fvdb_cpp

    case = next(item for item in _CASES if item[0] == args.case)
    name, kernel_size, stride, kind = case
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    candidate_rows = args.side**3
    for size in kernel_size:
        candidate_rows *= size
    # This is exact for full-support generated transpose; it is a safe preflight
    # upper bound for all other cases as well.
    if kind == "transpose" and candidate_rows > args.max_transpose_rows:
        raise ValueError(
            f"refusing {name} side={args.side}: {candidate_rows:,} generated rows exceed "
            f"--max-transpose-rows={args.max_transpose_rows:,}"
        )

    ijk = _dense_ijk(args.side, device)
    grid = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor(ijk), voxel_sizes=1.0, origins=0.0)
    _synchronize(device)
    cpu_before = _hwm_bytes() if device.type == "cpu" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
    else:
        allocated_before = reserved_before = 0

    start = time.perf_counter()
    output = (
        grid.conv_grid(kernel_size=kernel_size, stride=stride)
        if kind == "forward"
        else grid.conv_transpose_grid(kernel_size=kernel_size, stride=stride)
    )
    _synchronize(device)
    wall_time_ms = (time.perf_counter() - start) * 1000.0
    cpu_delta = max(0, _hwm_bytes() - cpu_before) if cpu_before is not None else None
    if device.type == "cuda":
        allocated_delta = max(0, torch.cuda.max_memory_allocated(device) - allocated_before)
        reserved_delta = max(0, torch.cuda.max_memory_reserved(device) - reserved_before)
    else:
        allocated_delta = reserved_delta = None

    stats = dict(_fvdb_cpp.last_conv_grid_resource_stats()) if kind == "forward" else None
    direct_cases = {"ks2_direct", "ks3_direct", "ks4_direct", "issue_668"}
    if name in direct_cases:
        if stats is None or not bool(stats["used_direct_projection"]):
            raise AssertionError(f"{name} did not use the required direct projection path")
        if int(stats["valid_emission_count"]) != grid.total_voxels:
            raise AssertionError(f"{name} must emit exactly one canonical row per input voxel")
        if int(stats["peak_requested_bytes"]) != 16 * grid.total_voxels:
            raise AssertionError(f"{name} direct staging must remain exactly one 16-byte row per input voxel")
    if "count_fill" in name:
        if stats is None or bool(stats["used_direct_projection"]):
            raise AssertionError(f"{name} did not use count-then-fill staging")
        emissions = int(stats["valid_emission_count"])
        if int(stats["emission_requested_bytes"]) != 16 * emissions:
            raise AssertionError(f"{name} emission allocation does not track exact M")
        if emissions >= candidate_rows:
            raise AssertionError(f"{name} is not a sparse-emission count-then-fill case")

    execution_median_ms = None
    if args.execution and kind == "forward":
        plan = fvdb.ConvolutionPlan.from_grid_batch(kernel_size=kernel_size, stride=stride, source_grid=grid)
        features = torch.ones((grid.total_voxels, 8), dtype=torch.float32, device=device)
        weights = torch.ones((8, 8, *kernel_size), dtype=torch.float32, device=device)
        plan.execute(features, weights)
        _synchronize(device)
        timings: list[float] = []
        for _ in range(args.execution_repetitions):
            start = time.perf_counter()
            plan.execute(features, weights)
            _synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
        execution_median_ms = sorted(timings)[len(timings) // 2]

    return Sample(
        case=name,
        kind=kind,
        device=str(device),
        side=args.side,
        input_voxels=grid.total_voxels,
        kernel_size=kernel_size,
        stride=stride,
        candidate_rows=candidate_rows,
        output_voxels=output.total_voxels,
        wall_time_ms=wall_time_ms,
        cpu_peak_delta_bytes=cpu_delta,
        cuda_peak_allocated_delta_bytes=allocated_delta,
        cuda_peak_reserved_delta_bytes=reserved_delta,
        final_grid_bytes=_grid_total_bytes(output),
        resource_stats=stats,
        execution_median_ms=execution_median_ms,
    )


def _worker_command(args: argparse.Namespace, case: str, side: int, device: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case",
        case,
        "--side",
        str(side),
        "--device",
        device,
        "--max-transpose-rows",
        str(args.max_transpose_rows),
        "--execution-repetitions",
        str(args.execution_repetitions),
    ]
    if args.execution:
        command.append("--execution")
    return command


def _installed_fvdb_path() -> str:
    import fvdb

    return str(Path(fvdb.__file__).resolve())


def _driver(args: argparse.Namespace) -> dict[str, Any]:
    source_package = (Path(__file__).resolve().parents[3] / "fvdb").resolve()
    imported_package = Path(_installed_fvdb_path()).parent
    if not args.allow_source_import and imported_package == source_package:
        raise RuntimeError("fVDB was imported from the source tree; run from the installed package environment")
    devices = ["cpu"]
    if torch.cuda.is_available() and not args.cpu_only:
        devices.append("cuda")
    records: list[dict[str, Any]] = []
    for device in devices:
        for side in args.sides:
            for case, _, _, _ in _CASES:
                proc = subprocess.run(
                    _worker_command(args, case, side, device),
                    check=True,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONPATH": ""},
                )
                records.append(json.loads(proc.stdout))
    import fvdb

    cuda_properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "fvdb_module": _installed_fvdb_path(),
        "fvdb_version": getattr(fvdb, "__version__", None),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "cuda_total_memory_bytes": int(cuda_properties.total_memory) if cuda_properties is not None else None,
        "sides": args.sides,
        "max_transpose_rows": args.max_transpose_rows,
        "records": records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=False, help="JSON output path (driver mode)")
    parser.add_argument("--sides", type=int, nargs="+", default=[16, 32, 48], help="geometric dense-grid side lengths")
    parser.add_argument("--cpu-only", action="store_true", help="skip CUDA even when available")
    parser.add_argument("--execution", action="store_true", help="also time plan execution for forward cases")
    parser.add_argument("--execution-repetitions", type=int, default=5)
    parser.add_argument("--max-transpose-rows", type=int, default=8_000_000)
    parser.add_argument(
        "--allow-source-import", action="store_true", help="test-only escape hatch for an uninstalled checkout"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=[case[0] for case in _CASES], help=argparse.SUPPRESS)
    parser.add_argument("--side", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execution_repetitions < 1 or args.max_transpose_rows < 1 or any(side < 1 for side in args.sides):
        parser.error("sizes, repetitions, and transpose cap must be positive")
    if args.worker and (args.case is None or args.side is None or args.device is None):
        parser.error("worker mode requires --case, --side, and --device")
    return args


def main() -> None:
    args = _parse_args()
    if args.worker:
        print(json.dumps(asdict(_run_worker(args)), sort_keys=True))
        return
    report = _driver(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
