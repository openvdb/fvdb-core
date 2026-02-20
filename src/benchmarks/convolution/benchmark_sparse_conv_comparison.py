# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Python-level benchmark comparing fVDB sparse convolution against other sparse
# convolution libraries (spconv, torchsparse, MinkowskiEngine) and dense PyTorch
# conv3d.  All libraries receive identical voxel coordinates and features so the
# comparison is apples-to-apples.
#
# Usage:
#   # Install optional competitors into the current conda env
#   python benchmark_sparse_conv_comparison.py --install-deps
#
#   # Run the full benchmark suite (benchmarks whatever is installed)
#   python benchmark_sparse_conv_comparison.py --output comparison_results.json
#
#   # Run a specific suite
#   python benchmark_sparse_conv_comparison.py --suite sparsity --output sparsity_results.json
#
#   # List available backends
#   python benchmark_sparse_conv_comparison.py --list-backends
#

from __future__ import annotations

import abc
import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------

_SPCONV_CUDA_VARIANTS = {
    "12.0": "spconv-cu120",
    "12.4": "spconv-cu124",
    "12.6": "spconv-cu126",
}


def _detect_spconv_package() -> str:
    """Pick the best spconv CUDA variant for the current environment."""
    if not torch.cuda.is_available():
        return "spconv"
    cuda_ver = torch.version.cuda or ""
    major_minor = ".".join(cuda_ver.split(".")[:2])
    best = "spconv-cu126"
    for ver, pkg in sorted(_SPCONV_CUDA_VARIANTS.items()):
        if major_minor >= ver:
            best = pkg
    return best


def install_deps(include_minkowski: bool = False) -> None:
    """pip-install optional competitor libraries into the current environment."""
    spconv_pkg = _detect_spconv_package()
    packages = [spconv_pkg]
    if include_minkowski:
        packages.append("git+https://github.com/NVIDIA/MinkowskiEngine")

    for pkg in packages:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg])


# ---------------------------------------------------------------------------
# Data generation helpers (deterministic, shared across all backends)
# ---------------------------------------------------------------------------


def generate_dense_coords(dim: int) -> torch.Tensor:
    """Return (N, 3) int32 ijk coordinates for a dense dim^3 grid."""
    r = torch.arange(dim, dtype=torch.int32)
    g = torch.meshgrid(r, r, r, indexing="ij")
    return torch.stack(g, dim=-1).reshape(-1, 3)


def generate_sparse_coords(bbox_dim: int, occupancy_pct: int, seed: int = 42) -> torch.Tensor:
    """Return (N, 3) int32 ijk coordinates at the given occupancy percentage."""
    total = bbox_dim**3
    n = max(1, total * occupancy_pct // 100)
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(total, generator=gen)[:n]
    ijk = torch.zeros(n, 3, dtype=torch.int32)
    ijk[:, 0] = (perm // (bbox_dim * bbox_dim)).to(torch.int32)
    ijk[:, 1] = ((perm // bbox_dim) % bbox_dim).to(torch.int32)
    ijk[:, 2] = (perm % bbox_dim).to(torch.int32)
    return ijk


# ---------------------------------------------------------------------------
# Abstract backend adapter
# ---------------------------------------------------------------------------


class BackendAdapter(abc.ABC):
    """Thin adapter translating common (ijk, features, weights) into a
    library-specific sparse convolution call."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @staticmethod
    @abc.abstractmethod
    def available() -> bool: ...

    @abc.abstractmethod
    def setup(
        self,
        ijk: torch.Tensor,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        device: torch.device,
        bbox_dim: int,
    ) -> None:
        """Build library-native structures.  *ijk* is (N, 3) int32 CPU."""
        ...

    @abc.abstractmethod
    def forward(self) -> torch.Tensor:
        """Run one forward convolution, return output tensor."""
        ...

    def teardown(self) -> None:
        """Release resources (optional)."""
        pass

    def setup_time_ms(self) -> float:
        """Return the time spent in setup(), measured by the harness."""
        return self._setup_time_ms

    @property
    def num_voxels(self) -> int:
        return self._num_voxels


# ---------------------------------------------------------------------------
# fVDB adapter
# ---------------------------------------------------------------------------


class FVDBAdapter(BackendAdapter):
    name = "fVDB"

    @staticmethod
    def available() -> bool:
        try:
            import fvdb  # noqa: F401

            return True
        except ImportError:
            return False

    def setup(self, ijk, in_channels, out_channels, kernel_size, device, bbox_dim):
        import fvdb

        self._num_voxels = ijk.shape[0]
        ijk_dev = ijk.to(device)
        jt = fvdb.JaggedTensor(ijk_dev)
        self._grid = fvdb.GridBatch.from_ijk(jt, voxel_sizes=1, origins=0, device=device)
        self._plan = fvdb.ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size,
            stride=1,
            source_grid=self._grid,
            target_grid=self._grid,
        )
        torch.manual_seed(0)
        n = self._grid.total_voxels
        self._features = torch.randn(n, in_channels, device=device)
        self._weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size, device=device)

    def forward(self):
        return self._plan.execute(self._features, self._weights)

    def teardown(self):
        del self._plan, self._grid, self._features, self._weights


# ---------------------------------------------------------------------------
# spconv adapter
# ---------------------------------------------------------------------------


class SpconvAdapter(BackendAdapter):
    name = "spconv"

    @staticmethod
    def available() -> bool:
        try:
            import spconv.pytorch  # noqa: F401

            return True
        except ImportError:
            return False

    def setup(self, ijk, in_channels, out_channels, kernel_size, device, bbox_dim):
        import spconv.pytorch as spconv

        self._num_voxels = ijk.shape[0]
        batch_idx = torch.zeros(ijk.shape[0], 1, dtype=torch.int32)
        indices = torch.cat([batch_idx, ijk], dim=1).to(device)
        torch.manual_seed(0)
        features = torch.randn(ijk.shape[0], in_channels, device=device)
        spatial_shape = [bbox_dim] * 3
        self._input = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size=1)
        self._conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key="bench").to(
            device
        )

    def forward(self):
        self._input.indice_dict = {}
        return self._conv(self._input).features

    def teardown(self):
        del self._conv, self._input


# ---------------------------------------------------------------------------
# torchsparse adapter
# ---------------------------------------------------------------------------


class TorchSparseAdapter(BackendAdapter):
    name = "torchsparse"

    @staticmethod
    def available() -> bool:
        try:
            import torchsparse  # noqa: F401

            return True
        except ImportError:
            return False

    def setup(self, ijk, in_channels, out_channels, kernel_size, device, bbox_dim):
        import torchsparse
        import torchsparse.nn as spnn

        self._num_voxels = ijk.shape[0]
        batch_idx = torch.zeros(ijk.shape[0], 1, dtype=torch.int32)
        coords = torch.cat([ijk, batch_idx], dim=1).int()
        torch.manual_seed(0)
        features = torch.randn(ijk.shape[0], in_channels, device=device)
        self._input = torchsparse.SparseTensor(feats=features, coords=coords.to(device))
        self._conv = spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False).to(device)

    def forward(self):
        return self._conv(self._input).feats

    def teardown(self):
        del self._conv, self._input


# ---------------------------------------------------------------------------
# MinkowskiEngine adapter
# ---------------------------------------------------------------------------


class MinkowskiAdapter(BackendAdapter):
    name = "MinkowskiEngine"

    @staticmethod
    def available() -> bool:
        try:
            import MinkowskiEngine  # noqa: F401

            return True
        except ImportError:
            return False

    def setup(self, ijk, in_channels, out_channels, kernel_size, device, bbox_dim):
        import MinkowskiEngine as ME

        self._num_voxels = ijk.shape[0]
        batch_idx = torch.zeros(ijk.shape[0], 1, dtype=torch.int32)
        coords = torch.cat([batch_idx, ijk], dim=1).int()
        torch.manual_seed(0)
        features = torch.randn(ijk.shape[0], in_channels, device=device)
        self._input = ME.SparseTensor(features=features, coordinates=coords, device=device)
        self._conv = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
            dimension=3,
        ).to(device)

    def forward(self):
        return self._conv(self._input).F

    def teardown(self):
        del self._conv, self._input


# ---------------------------------------------------------------------------
# Dense PyTorch conv3d adapter
# ---------------------------------------------------------------------------


class DenseAdapter(BackendAdapter):
    name = "Dense (conv3d)"

    @staticmethod
    def available() -> bool:
        return True

    def setup(self, ijk, in_channels, out_channels, kernel_size, device, bbox_dim):
        self._num_voxels = bbox_dim**3
        torch.manual_seed(0)
        self._input = torch.randn(1, in_channels, bbox_dim, bbox_dim, bbox_dim, device=device)
        self._weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size, device=device)
        self._padding = kernel_size // 2

    def forward(self):
        return F.conv3d(self._input, self._weight, padding=self._padding)

    def teardown(self):
        del self._input, self._weight


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_ADAPTERS: list[type[BackendAdapter]] = [
    FVDBAdapter,
    SpconvAdapter,
    TorchSparseAdapter,
    MinkowskiAdapter,
    DenseAdapter,
]


def get_available_adapters() -> list[type[BackendAdapter]]:
    return [a for a in ALL_ADAPTERS if a.available()]


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    library: str
    suite: str
    params: dict[str, Any]
    num_voxels: int
    setup_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    num_iters: int
    times_ms: list[float] = field(default_factory=list)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_one(
    adapter_cls: type[BackendAdapter],
    ijk: torch.Tensor,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    device: torch.device,
    bbox_dim: int,
    suite_name: str,
    params: dict[str, Any],
    warmup: int = 3,
    num_iters: int = 20,
) -> BenchmarkResult | None:
    adapter = adapter_cls()
    try:
        _sync()
        t0 = time.perf_counter()
        adapter.setup(ijk, in_channels, out_channels, kernel_size, device, bbox_dim)
        _sync()
        setup_ms = (time.perf_counter() - t0) * 1e3
    except Exception as e:
        print(f"  [{adapter.name}] setup failed: {e}")
        return None

    try:
        for _ in range(warmup):
            adapter.forward()
            _sync()

        times: list[float] = []
        for _ in range(num_iters):
            _sync()
            t0 = time.perf_counter()
            adapter.forward()
            _sync()
            times.append((time.perf_counter() - t0) * 1e3)

        import numpy as np

        arr = np.array(times)
        result = BenchmarkResult(
            library=adapter.name,
            suite=suite_name,
            params=params,
            num_voxels=adapter.num_voxels,
            setup_ms=setup_ms,
            mean_ms=float(arr.mean()),
            std_ms=float(arr.std()),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
            num_iters=num_iters,
            times_ms=[float(t) for t in times],
        )
    except Exception as e:
        print(f"  [{adapter.name}] forward failed: {e}")
        return None
    finally:
        adapter.teardown()
        _sync()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


def suite_grid_size(
    adapters: list[type[BackendAdapter]],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Sweep dense grid sizes at fixed C=32, K=3."""
    results: list[BenchmarkResult] = []
    dims = [8, 16, 24, 32]
    C = 32
    K = 3
    for dim in dims:
        ijk = generate_dense_coords(dim)
        params = {"grid_dim": dim, "voxels": dim**3, "channels": C, "kernel_size": K}
        print(f"\n[grid_size] dim={dim} ({dim**3} voxels), C={C}, K={K}")
        for acls in adapters:
            r = benchmark_one(acls, ijk, C, C, K, device, dim, "grid_size", params, warmup, num_iters)
            if r:
                print(f"  {r.library:20s}  mean={r.mean_ms:8.3f} ms  std={r.std_ms:6.3f}  setup={r.setup_ms:8.3f} ms")
                results.append(r)
    return results


def suite_sparsity(
    adapters: list[type[BackendAdapter]],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Sweep occupancy at fixed channel width, multiple bbox sizes."""
    results: list[BenchmarkResult] = []
    configs = [
        (64, [1, 5, 10, 25, 50, 100]),
        (128, [1, 5, 10, 25, 50]),
        (256, [1, 5, 10, 25]),
    ]
    C = 32
    K = 3
    for bbox_dim, occupancies in configs:
        for occ in occupancies:
            ijk = generate_dense_coords(bbox_dim) if occ >= 100 else generate_sparse_coords(bbox_dim, occ)
            n_voxels = ijk.shape[0]
            params = {
                "bbox_dim": bbox_dim,
                "occupancy_pct": occ,
                "voxels": n_voxels,
                "channels": C,
                "kernel_size": K,
            }
            print(f"\n[sparsity] bbox={bbox_dim}, occ={occ}% ({n_voxels} voxels), C={C}, K={K}")
            for acls in adapters:
                # Skip dense baseline at occupancies < 100 -- it always uses the full grid
                if acls is DenseAdapter and occ < 100:
                    r = benchmark_one(acls, ijk, C, C, K, device, bbox_dim, "sparsity", params, warmup, num_iters)
                elif acls is DenseAdapter:
                    r = benchmark_one(acls, ijk, C, C, K, device, bbox_dim, "sparsity", params, warmup, num_iters)
                else:
                    r = benchmark_one(acls, ijk, C, C, K, device, bbox_dim, "sparsity", params, warmup, num_iters)
                if r:
                    print(
                        f"  {r.library:20s}  mean={r.mean_ms:8.3f} ms  std={r.std_ms:6.3f}"
                        f"  setup={r.setup_ms:8.3f} ms"
                    )
                    results.append(r)
    return results


def suite_channels(
    adapters: list[type[BackendAdapter]],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Sweep channel width at fixed 16^3 dense grid, K=3."""
    results: list[BenchmarkResult] = []
    dim = 16
    K = 3
    channels = [4, 16, 32, 64, 128, 256]
    ijk = generate_dense_coords(dim)
    for C in channels:
        params = {"grid_dim": dim, "voxels": dim**3, "channels": C, "kernel_size": K}
        print(f"\n[channels] dim={dim}, C={C}, K={K}")
        for acls in adapters:
            r = benchmark_one(acls, ijk, C, C, K, device, dim, "channels", params, warmup, num_iters)
            if r:
                print(f"  {r.library:20s}  mean={r.mean_ms:8.3f} ms  std={r.std_ms:6.3f}  setup={r.setup_ms:8.3f} ms")
                results.append(r)
    return results


SUITES = {
    "grid_size": suite_grid_size,
    "sparsity": suite_sparsity,
    "channels": suite_channels,
}

# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _env_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "N/A",
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return meta


def save_results(results: list[BenchmarkResult], path: str) -> None:
    data = {
        "metadata": _env_metadata(),
        "benchmarks": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fVDB sparse convolution against alternative libraries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--install-deps", action="store_true", help="pip-install optional competitor libraries")
    parser.add_argument(
        "--install-minkowski", action="store_true", help="Also attempt to install MinkowskiEngine (fragile)"
    )
    parser.add_argument("--list-backends", action="store_true", help="List available backends and exit")
    parser.add_argument(
        "--suite",
        nargs="*",
        choices=list(SUITES.keys()),
        default=None,
        help="Benchmark suites to run (default: all)",
    )
    parser.add_argument("--output", "-o", type=str, default="comparison_results.json", help="Output JSON path")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations (default: 20)")
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Restrict to these backends (by name substring, e.g. 'fVDB spconv')",
    )
    args = parser.parse_args()

    if args.install_deps:
        install_deps(include_minkowski=args.install_minkowski)
        print("Done installing dependencies.")
        return

    available = get_available_adapters()

    if args.list_backends:
        print("Available backends:")
        for a in available:
            print(f"  - {a.name}")  # type: ignore[attr-defined]
        missing = [a for a in ALL_ADAPTERS if not a.available()]
        if missing:
            print("Not available (install with --install-deps):")
            for a in missing:
                print(f"  - {a.name}")  # type: ignore[attr-defined]
        return

    if args.backends:
        filtered = []
        for a in available:
            if any(b.lower() in a.name.lower() for b in args.backends):  # type: ignore[attr-defined]
                filtered.append(a)
        available = filtered

    if not available:
        print("No backends available. Run with --install-deps or check your fVDB installation.")
        sys.exit(1)

    print("Backends under test:")
    for a in available:
        print(f"  - {a.name}")  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    suites_to_run = args.suite if args.suite else list(SUITES.keys())
    all_results: list[BenchmarkResult] = []

    for suite_name in suites_to_run:
        print(f"\n{'='*60}")
        print(f"  Suite: {suite_name}")
        print(f"{'='*60}")
        suite_fn = SUITES[suite_name]
        results = suite_fn(available, device, args.warmup, args.iters)
        all_results.extend(results)

    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
