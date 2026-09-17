fVDB Version History
====================

## Version 0.6.0 - In Development

*91 commits, 259 files changed, 5 contributors.*

This release unifies sparse convolution geometry, reduces grid-construction memory and batch overhead, moves
Gaussian splatting's high-level API to fVDB Reality Capture, and adds native volume rendering and interactive
viewer controls in the Viewer. It also migrates C++ grid storage to NanoVDB's single-space memory-resource API.

**Contributors:** @swahtz, @matthewdcong, @harrism, @phapalova, @blackencino

### Compatibility & Distribution

- **PyTorch 2.13:** fVDB updated to build, test and publish with PyTorch 2.13, CUDA 13.0/13.2 and Python 3.10-3.15
  support. CUDA 12.8 is no longer in the wheel build/test matrix (#699, #738).
- Removed compatibility code for PyTorch versions older than 2.6; package metadata now requires `torch>=2.7`.
  Prebuilt wheels must still match their advertised PyTorch/CUDA versions (#700).
- Added a `StrEnum` compatibility wrapper so the convolution enums work on Python 3.10 (#740).
- Added weekly compatibility testing against the oldest PyTorch supporting the lowest supported CUDA version,
  currently PyTorch 2.9.1 with CUDA 13.0 (#745).

### SDF & Ray Traversal

- **Breaking:** Renamed `retopologize_sdf` to `rebuild_narrow_band` across `Grid`, `GridBatch`, and
  `fvdb.functional`; no compatibility alias is provided (#762).
- Fixed `reinitialize_sdf` creating phantom interior interfaces in narrow-band SDFs. Inactive neighbors and
  newly padded voxels now inherit the adjacent active voxel's sign (#762).
- **Input requirements:** Occupancy-to-SDF inputs must include a positive exterior layer; all-negative inputs
  produce an empty narrow band. Both SDF operations require finite scalar values shaped `(N,)` or `(N, 1)`,
  and `reinitialize_sdf` rejects anisotropic voxels (#762).
- Fixed spurious near-origin hits in `ray_implicit_intersection` when active TSDF voxels contain zero-valued
  no-data samples. Exact zeros now act as gaps instead of seeding or triggering a sign crossing (#698).
- Accelerated HDDA ray traversal with NanoVDB's fused `ReadAccessor::getDimAndActive`: `voxels_along_rays`
  measured 1.36-1.68x faster, and segment, uniform-sampling, and implicit-intersection operations 1.03-1.30x
  faster, on NanoVDB example grids with 512x512 rays on RTX PRO 6000 Blackwell (#754).

### Sparse Convolution Semantics & Migration

The following changes implement the canonical convolution contract in PR #726 (issue #668).

- **Breaking:** Unified sparse convolution and transposed-convolution geometry around the componentwise Torch-phase
  relation ``fine_ijk = stride * coarse_ijk + tap_ijk - padding_before``, where
  ``padding_before = floor((kernel_size - 1) / 2)`` and each zero-based tap component satisfies
  ``0 <= tap_ijk[axis] < kernel_size[axis]`` (issue #668). Generated topology now uses the `complete` policy; an
  explicit target uses the `restricted` policy and is an evaluation domain, not a different kernel rule. This changes
  generated topology for affected even kernels, `kernel_size == stride >= 3`, and `kernel_size=1, stride>1`; the
  proved `kernel_size=stride=2` path is retained.
- **Breaking:** Generated strided convolution grids now have voxel size `stride * h` and the canonical input
  origin; generated transposed grids apply the inverse scale. Explicit convolution grid pairs with incompatible
  scale or origin registration now fail instead of being silently interpreted in index space. `coarsened_grid`
  remains a block-centroid cell grid and must not be used as a convolution target.
- `ConvolutionPlan.from_grid_batch_transposed(..., target_grid=None)` now generates full transposed support.
  A saved decoder target is an explicit restriction and may have zero-degree rows; transposed convolution is
  adjoint connectivity, not a value inverse. `ConvolutionPlan.from_plan_transposed` reverses a plan's exact
  finite edges; a tied weighted adjoint uses `weight.transpose(0, 1).contiguous()`.
- `Grid.conv_grid(1, 1)`, `GridBatch.conv_grid(1, 1)`, and their transposed-grid counterparts now return the
  source object, preserving the identity matmul path and its compact two-dimensional weight layout. One-tap
  convolutions with larger stride sample stride-aligned residues rather than coarsening all blocks. Removed rows
  had zero linear degree before bias, but a model with bias can observe their removal.
- Coverage histograms are computed lazily and shared with exact plan transposes. Generated-support and
  strict-target checks remain eager; CUDA transposed-convolution allocation failures now include the required
  staging size.
- Added typed convolution topology/phase policies, topology provenance, coverage diagnostics and warnings,
  and builder resource statistics.
- **Migration:** Invalidate topology/plan caches and include the convolution-semantics version in cache keys.
  Weight ordering is unchanged, but affected checkpoints may produce different topology and world registration.
  No legacy geometry mode is provided.
- The dense convolution expert backend is disabled until it can realize this contract exactly. PredGatherIGemm
  remains limited to its supported odd kernels and strides; unsupported combinations fail at plan construction.

### Gaussian Splatting API Migration

- **Breaking:** Moved the high-level Gaussian splatting Python API to fVDB Reality Capture and removed its former
  `fvdb` entry points. Use `fvdb_reality_capture.GaussianSplat3d`,
  `fvdb_reality_capture.ProjectedGaussianSplats`, `fvdb_reality_capture.gaussian_render_jagged`, and
  `fvdb_reality_capture.evaluate_spherical_harmonics` instead. The associated `ShOrderingMode`,
  `RollingShutterType`, `CameraModel`, and `ProjectionMethod` enums now also live in `fvdb_reality_capture`. The
  low-level compiled Gaussian kernels and viewer support remain in fVDB Core (#685).
- Decoupled `fvdb.viz` from application-owned Gaussian model classes. `Scene.add_gaussian_splat_3d` now accepts the
  immutable, core-owned `GaussianSplatViewData` tensor contract and delegates to the new
  `Scene.add_gaussian_splat_tensors` primitive. The legacy six-property model interface remains temporarily available
  with a deprecation warning (#685).

### Grid Construction & Topology

Performance figures below and in the rendering section are PR-reported measurements against each change's
pre-change baseline; they are workload-specific and are not cumulative release-to-release speedups.

- Replaced expanded coordinate-list construction with NanoVDB leaf-mask morphology for CUDA dual/padded grids
  and supported convolution, refinement, coarsening, and merge paths, avoiding expanded candidate-list int32
  overflow. On an 11.5M-output-voxel dense-cube benchmark, `conv_grid(k3, s1)` construction fell from 246 to
  1.85 ms (~133x faster), with temporary memory dropping from 11.23 GB to 8.6 MB on RTX PRO 6000 Blackwell
  (#710, #712).
- Batched supported CUDA refinement, coarsening, and generated convolution topology construction to reduce
  per-grid launches and synchronization. Added an identity-convolution-plan fast path and reduced metadata
  overhead. For 48 synthetic shell grids, `conv_transpose_grid(k2, s2)` construction fell from 25.6 to 0.94 ms
  (~27x faster) on RTX PRO 6000 Blackwell (#757).
- CUDA point-grid construction now transforms points to voxel coordinates on demand, eliminating the temporary
  `(N, 3)` int32 coordinate tensor (12 bytes per input point), including for strided point inputs. Seven
  point-cloud benchmarks on RTX PRO 6000 Blackwell showed 1.7-10.2% lower execution time; peak memory was
  unchanged in the reported memory tests (#719).
- Morton and Hilbert encoders now use the stored batch bounding-box minimum for their default offset instead
  of materializing and reducing every voxel coordinate. `GridBatch.morton()` measured 1.67-1.79x faster on
  400K-5M voxels on an RTX PRO 6000 Blackwell GPU (#748).
- Fixed topology operations on sliced/indexed `GridBatch` views selecting physical parent grids instead of the
  requested logical grids; extended the fix to injection, merged grids, PredGatherIGemm convolution, SDF
  reinitialization, and NanoVDB export (#712, #787).
- Fixed CPU `pruned_grid` failing when a batch contains a single-voxel grid (#715), and updated NanoVDB to fix
  distributed multi-GPU grid building for the single-voxel case (#735).
- Generic padding now validates `bmin <= 0 <= bmax` and explicitly rejects `exclude_border` on PrivateUse1
  instead of reaching an unsupported path (#710).

### C++ Grid Storage, Allocation & I/O

- **Breaking (C++):** `GridBatchData` now uses private `GridStorage` backed by NanoVDB's single-space buffers.
  Replace `nanoGridHandle()` with logical `hostGridPtrAt` / `deviceGridPtrAt` accessors, order reads after
  `storageStream()`, and use `fvdb::detail::ops::contiguousGridStorage` for batch copies. `TorchDeviceBuffer`
  and `TorchStorageResource` are deprecated for one release; the latter moved to `TorchDeviceBuffer.h`.
  Builders no longer compute per-grid checksums (#770; PRs #773, #786, #787, #788, #790).
- NanoVDB builder scratch now uses PyTorch's active CUDA allocator, sharing its memory pool with tensors and
  respecting allocator configuration or replacement. Device grid storage uses `TorchDeviceResource`, keyed to
  its retained stream; scratch uses `TorchResource`, keyed to the operation's stream (#732, #773, #786, #788).
- Unified compaction, concatenation, serialization, and builder output assembly. Cross-device copies of sliced
  batches compact directly onto the destination; replicated dense grids and empty batch members reuse source
  storage during assembly. CUDA builders now run on the current PyTorch stream (#787, #788).
- Fixed serialization of sliced and empty batches, CUDA metadata version initialization, and NanoVDB export
  synchronization and header consistency (#773, #787, #788, #790).
- Fixed a host-metadata allocation leak for PrivateUse1 grids (#788).

### Neural Networks & Examples

- Added `fvdb.nn.Prune` to prune grid topology and aligned features with a per-voxel boolean mask, preserving
  gradients through retained feature rows (#753).
- Added generative shape-completion and shape-VAE examples using generated transposed convolutions and pruning,
  a shape-VAE notebook with latent sampling/interpolation, and a `load_gso_shoes()` example-data loader (#753).

### Viewer

- Added `Scene.add_level_set` and `Scene.add_fog_volume` for native SDF and fog rendering, with one view per grid
  in a batch (#690, #694).
- Added slider, number, text, and checkbox widgets with Python/C++ value access, callbacks, scene cleanup, and
  viewer shutdown support (#649).
- Raised the optional `nanovdb-editor` minimum to 0.1.6 and restricted C++ includes to its public headers
  (#694, #702, #693).

### Gaussian Kernels, Rendering & PLY I/O

These improvements apply to the low-level kernels retained in fVDB Core and used by fVDB Reality Capture.

- Added multi-GPU Unscented Transform projection and world-space rasterization for 3DGUT-style training with
  distorted camera models; also reduced single-GPU UT projection shared-memory usage (#729).
- Replaced bounding-box-only Gaussian/tile intersection tests with ellipse-tile tests, reducing intersection
  records by 30-40%, with reported performance improvements of approximately 3% on one GPU and 7% on two GPUs
  (#742).
- Expanded Gaussian intersection offsets and indices to 64 bits to prevent overflow, invalid memory access,
  and corrupt training results on large scenes (#708).
- Fused view-direction computation into spherical harmonics evaluation and reduced backward camera-gradient
  overhead using device-local block sums and segmented reductions. SH backward operation time fell by
  23.5-57.5% for 1M-100M Gaussians and 1-4 cameras on dual RTX 3090s (#687, #697, #751).
- Reduced multi-GPU traffic and atomic contention with device-local intersection keys, rasterization and
  projection gradient buffers, and NCCL reductions. Overlapped destination prefetch with gradient computation
  and reduction. Projection backward measured 1.43-2.18x faster on dual RTX 3090s (#703, #713, #749, #783).
- Fixed a multi-GPU backward-rasterization race where one GPU could clear another GPU's accumulated gradients
  (#706), and guarded SH/projection work on GPUs assigned zero Gaussians (#728).
- Fixed fused SSIM prefetch using NHWC offsets for NCHW tensors: measured speedups ranged from 1.76-4.67x
  across 1080p-8K inference, training, and backward benchmarks on dual RTX 3090s (#750). Corrected CUDA event
  creation to honor the requested flags (#686).
- Reduced forward-rasterization compile time, stack usage, and generated code size by matching output storage
  to the channel chunk size: compilation fell from 616 to 181 seconds (~71% less time), and generated GPU code
  shrank from 13.7 to 8.3 MB in the reported build (#714). Dense-mode contributing-Gaussian repacks now use
  `view()` to enforce their no-copy layout assumption (#717).
- Updated tinyply to 3.0: a 375M-Gaussian SH3 PLY loaded in 4 minutes instead of 12 (~3x faster) in the reported
  test (#744). Added CPU Gaussian NaN/Inf-mask computation so CPU PLY export no longer needs a GPU round trip
  for validation (#765).

### Build, Packaging, Documentation & CI

- Added `docker/build_wheel.py` for production-style wheels from a local checkout with selectable
  Python/PyTorch/CUDA versions and GPU architectures; release and nightly publishing share this build recipe
  (#731).
- Added an `NVCC_THREADS` environment override to tune compilation parallelism (#704), plus `./build.sh format`
  and `./build.sh format check` for the repository's C++ and Python formatting rules (#769).
- Added the `fvdb-core[examples]` extra and lazy loading of optional example dependencies. Test/example data
  now downloads pinned source snapshots without GitPython; `fvdb.utils.tests` remains available for downstream
  repositories, and GitPython remains in development/test environments (#718, #727, #730).
- Completed the public Python API reference and added CI checks for docstrings, parameter descriptions, and
  100% coverage of the configured public API inventory (#784).
- Clarified installation and hardware-support guidance, separated stable-release install commands from nightly
  dependency versions, and automated stable documentation metadata updates from release tags (#724, #731, #738).
- Enforced matching Python/CUDA/PyTorch pins across conda environments (#756), and reduced unit-test GPU memory
  requirements (#701).
- Improved CI runner provisioning with availability-zone fallbacks, shared retry/backoff workflows, SM 8.6
  fallback support, matching test-wheel architectures, and cleanup after cancellation
  (#705, #760, #763, #768, #789). Tightened runner-token forwarding policy and updated pinned workflow actions
  (#696, #723, #759).
- Fixed nightly publish environment setup and reused published wheel hashes to avoid redundant downloads
  during index generation (#695, #774).

## Version 0.5.1 - July 16, 2026

**Bug Fixes:**

- Fixed a reference cycle in `_InjectFn` causing a memory leak (#688).

## Version 0.5.0 - July 1, 2026

*115 commits, 500+ files changed, 10 contributors.*

This release eliminates the C++ implementation behind `GridBatch` in favor of a new `torch.nn.functional`-style `fvdb.functional` module, with `GridBatch` now a thin pure-Python class delegating to it, and adds a complementary single-grid `Grid` class alongside it. It also moves the entire Gaussian splatting autograd and rendering pipeline from C++ into pure Python. Multi-GPU Gaussian splatting rasterization and projection get another round of performance tuning, and fVDB gains its first SDF reinitialization/retopologization operators alongside NanoVDB loading fixes and PyTorch 2.11/Python 3.14 support. Documentation has moved to a fully versioned Read the Docs site.

**Highlights:**
- Eliminated the C++ `GridBatch` implementation for a new functional API (`fvdb.functional`), with `GridBatch` reimplemented as a thin Python wrapper and a new complementary `Grid` class added, and moved Gaussian splatting's autograd/pipeline logic from C++ to pure Python.
- Continued multi-GPU Gaussian splatting performance work: repartitioned SH/projection kernels, smarter prefetching, shared-memory rasterization optimizations, and improved parity with gsplat.
- Added the first SDF reinitialization/retopologization operators and a `sample_nearest` grid sampling operator, and generalized `volume_render` to N channels.
- Fixed NanoVDB loading of mixed grid types and added a `read_metadata` API; optimized `saveNVDB`/`to_nanovdb`.
- fVDB now supports PyTorch 2.11, Python 3.14, and SM 8.6, with the `torch_scatter` and `torchsparse` dependencies removed.
- Documentation moved to a fully versioned Read the Docs site, with new interactive TEACHME lessons for the core API.

**Contributors:** @swahtz, @matthewdcong, @harrism, @fwilliams, @phapalova, @blackencino, @areidmeyer, @zlalena, @mvanhorn, @jinhwanlazy

---

### Core Library Architecture (Major)

- Eliminated the legacy C++ `GridBatch` wrapper class and its `Vec3*OrScalar` type system (~2,600 lines), replacing them with a frozen `GridBatchData` struct and a `torch.nn.functional`-style `fvdb.functional` module (~120 functions with explicit `_batch`/`_single` variants). `GridBatch` is now a thin pure-Python class whose methods delegate to `fvdb.functional`, and a new complementary `Grid` class was added for the single-grid case (`grid_count == 1`), with plain `torch.Tensor` I/O instead of `JaggedTensor` (#582 - @blackencino).
- Promoted `GridBatchData`, `TorchDeviceBuffer`, `VoxelCoordTransform`, and related type traits from private `detail` headers to the public `fvdb` namespace, and added CMake installation rules for public headers so downstream C++ projects can link against fVDB (#632, #633 - @swahtz).

---

### Gaussian Splatting & Rendering

**New Features:**
- Updated the Gaussian splatting camera API to separate camera semantics from projection implementation, replacing the old `ProjectionType`/`DistortionModel` split with explicit `CameraModel`/`ProjectionMethod` controls, and added world-space render parity for depth and RGBD paths (#518 - @fwilliams).

**Architecture:**
- Moved the entire Gaussian splatting autograd and pipeline from C++ into pure Python `torch.autograd.Function` implementations, eliminating the C++ `GaussianSplat3d` class (#595 - @fwilliams), with a follow-up fixing a synchronization regression it introduced (#599 - @matthewdcong) and restoring comments lost in the migration (#603 - @swahtz).
- Renamed 21 Gaussian splatting operator file pairs for clarity and extracted shared CUDA utilities (BinSearch, CubWrapper, Prefetch, WarpReduce) with no functional changes (#596 - @fwilliams).

**Optimizations:**
- Repartitioned multi-GPU spherical harmonics and projection kernels across Gaussians instead of camera-Gaussian pairs, removing the system-scope atomics that bottlenecked the interconnect (#546, #547 - @matthewdcong).
- Added shared-memory feature caching and opacity-threshold culling to forward/backward rasterization, and reduced shared memory usage in pinhole projection by parameterizing blocks per-camera (#554, #555 - @matthewdcong).
- Fused `computeGradientState` into `projectionBackwardKernel` to avoid redundant global memory reads (#560 - @matthewdcong).
- Improved multi-GPU prefetching with batched `cudaMemPrefetchBatchAsync` calls and finer per-tile (rather than per-camera) granularity, and iterated multi-GPU tile intersection to reduce cross-device communication during the radix sort (#657, #664, #665 - @matthewdcong).
- Added a warp-level early exit to forward rasterization on top of the existing block-level exit (#658 - @matthewdcong).
- Improved rasterization parity with gsplat by tuning the alpha threshold and sigma clamp, and using per-axis 2D radii for tile intersection (#659 - @matthewdcong).
- Reordered `Gaussian2D` fields for better memory alignment, removed an unused spherical harmonics function, and eliminated a redundant delta computation (#624, #630, #651 - @matthewdcong).
- Materialized repeated opacities so image-space multi-GPU and world-space rasterization implementations always see a contiguous tensor (#600 - @matthewdcong).

**Bug Fixes:**
- Fixed a missing warp-level reduction causing incorrect quaternion gradient accumulation across multiple cameras in the projection backward pass (#534 - @swahtz).
- Fixed a `cudaErrorIllegalArgument` crash in tile intersection prefetch when a subset of Gaussians has zero intersections (#553 - @matthewdcong).
- Fixed gradient accumulation tensors not being initialized on the Unscented Transform projection's early-return path (#608 - @harrism).

---

### Grid Operations, Sampling & SDF

- Added a `sample_nearest` nearest-neighbor sampling operator for `GridBatch` and `Grid` (#628, #637 - @swahtz).
- Added Vec2 and double-precision vectorized fast paths to `SampleGridTrilinear` (#639 - @swahtz).
- Generalized `volume_render` to support up to 16 radiance channels for spectral rendering pipelines, rewrote its forward pass around per-ray register accumulation, and moved its autograd layer to Python (#636, #640 - @swahtz).
- Added narrow-band SDF reinitialize and retopologize operators built around a ported VoxelBlockManager eikonal solver — the first component of a NanoVDB port of OpenVDB's `VolumeToMesh` (#669 - @swahtz).
- Optimized the per-ray SDF zero-crossing kernel `ray_implicit_intersection` and its underlying HDDA traversal layer, fixing several correctness issues along the way (#663 - @swahtz).
- Optimized `saveNVDB`/`to_nanovdb` to build grids directly on-device (or via a host-only fast path) instead of rebuilding through CPU `setValue` calls (#650 - @swahtz).
- Fixed a CUDA crash in `Grid.inject_from()` when the source grid has 0 active voxels, which surfaced in narrow-band level-set simulations (#616 - @harrism).
- Fixed the TSDF/feature-blending weighted average formula to correctly apply `pixelWeight` to new samples (#588 - @jinhwanlazy).

---

### NanoVDB

- Fixed wrong-type dispatch when loading multi-grid NanoVDB handles with mixed grid types, and added a `read_metadata` API (#641 - @swahtz).
- Fixed TensorGrid blind-data loading reading channel 0 for every channel, which silently returned incorrect data for all channels of multi-channel grids (#652 - @mvanhorn).
- Updated the bundled NanoVDB Editor dependency, moving off a version with ABI issues (#556 - @areidmeyer).
- Added `nanovdb-editor` as an optional dependency and switched to consuming it from pip instead of building it from source (#559, #580, #581 - @swahtz, @phapalova).

---

### JaggedTensor

- Reimplemented `jsum`/`jmin`/`jmax` reductions on top of PyTorch's built-in `scatter_reduce_`, removing ~450 lines of custom CUDA/autograd code and the `torch_scatter` test dependency it required (#578, #571 - @swahtz).

---

### Neural Network Modules

- Added support for additional PyTorch attention backends (Flash, memory-efficient, math) in `scaled_dot_product_attention` by wrapping JaggedTensor data in nested tensors, selected the same way as PyTorch's built-in `sdpa_kernel` context manager (#365 - @swahtz).

---

### Core Library: Correctness & Performance

- Eliminated redundant `.item()` calls on CUDA tensors that triggered implicit device syncs, consolidating them into single bulk `.cpu()` transfers (#586 - @swahtz).
- Passed the current PyTorch CUDA stream to 15 kernel launches that previously used the implicit default stream, and added missing CUDA device guards across kernel-launching functions (#587, #589 - @swahtz).
- Replaced the runtime `numThreads` argument in the forEach CUDA dispatch framework with a compile-time template parameter, enabling `__launch_bounds__` on all forEach kernels for better register allocation (#638 - @swahtz).

---

### Viewer

- Added a `camera_fov` getter/setter to `fvdb.viz.Scene`, exposed through both C++ and Python (#558 - @swahtz).
- Fixed `fvdb.viz.PointCloudView` using an outdated `add_gaussian_splat_3d_view` signature (#631 - @swahtz).

---

### PyTorch & CUDA Compatibility

- fVDB now builds and runs with **PyTorch 2.11** and adds **Python 3.14** support, including SM 8.6 in the published wheels' CUDA architecture list to match PyTorch's support (#573, #561 - @swahtz, @matthewdcong).
- Removed the `torchsparse` dependency from all environment and CI configurations (#572 - @swahtz).

---

### Build & Packaging

- Upgraded clang-tools to 21 to fix a clangd SIGSEGV on CUDA files, and conda environments to gcc/g++ 14.3 (#491, #557 - @fwilliams, @swahtz).
- Removed the vestigial `setup.py` and GitLab CI configuration (#570 - @swahtz).
- Refactored CMake to consume libtorch through the canonical `torch` imported target instead of legacy `TORCH_INCLUDE_DIRS`/`TORCH_LIBRARIES` variables, retiring the deprecated THC headers and fixing a Conda build failure along the way (#661, #662 - @matthewdcong, @swahtz).
- Fixed a Torch CMake header path issue and disambiguated CI job names (#635 - @swahtz).
- Sped up incremental builds with ccache/sccache auto-detection, host-side precompiled headers, and trimmed Torch header includes (#644 - @swahtz).
- Updated the dev environment's OpenUSD version (#667 - @zlalena).

---

### Documentation

- Migrated documentation hosting to Read the Docs with full versioned-docs support: a pre-build hook for version generation, a dedicated Sphinx-build CI workflow, and sidebar/redirect/URL fixes (#610, #613, #615, #618, #622, #623, #625, #646 - @swahtz).
- Added `docs/TEACHME`, a set of LLM-loadable interactive lesson documents that teach the fVDB API through an AI coding assistant (#584 - @harrism).
- Fixed and expanded the tutorial notebooks — moved them out of WIP status, corrected broken API calls, added CI testing, and incorporated review feedback (#592, #598 - @harrism).
- Fixed the docs deployment workflow, Sphinx code-sample borders, and README/docs redirect URLs (#566, #577, #605, #626 - @swahtz).
- Fixed the `marching_cubes` docstrings to describe the third return value correctly as unique vertex indices (int64), not vertex normals (#653 - @mvanhorn).

---

### CI / DevOps / Release Infrastructure

- Hardened the release process scripts: idempotent `start-release.sh` re-runs, draft release PRs with fixed smoke-test Python setup, branch-integrity preservation in `finish-release-process.sh`, and automated doc-version updates (#528, #529, #544, #552, #563 - @harrism, @swahtz).
- Fixed the publish workflow across several iterations: Rocky Linux 8 / manylinux_2_28 containers, the Python install action, additional system dependencies, and dual S3 + PyPI publishing with GPU-validated tests (#536, #537, #538, #540, #545 - @harrism, @swahtz).
- Fixed several nightly wheel build and publish issues: missing tool errors, stale caches, CloudFront invalidation, and anchoring the nightly version to the upcoming release in `pyproject.toml` (#549, #574, #634, #645, #647 - @swahtz, @phapalova).
- Replaced the Slack unanswered-issues report with event-driven issue triage labels, after first fixing its insider-issue filtering (#522, #551 - @harrism).
- Centralized GitHub workflow and doc-version configuration into shared config, updated CI Actions versions, added git installation to CI system dependencies, and reverted a failing drop-cache step (#569, #611, #526, #629 - @swahtz, @phapalova).
- Swept CI tokens to least-privilege scopes and scoped the bundled shellcheck security check to real issues (#672, #674 - @swahtz).
- Removed the PyTorch upper-bound pin in `pyproject.toml`, updated `CONDA_OVERRIDE_CUDA` to 13.0, fixed a flaky bfloat16 JaggedTensor test, and silenced spurious warnings in the test suite (#671, #519, #517, #654 - @swahtz, @fwilliams, @mvanhorn).

---

### Repository Governance

- Split `CODEOWNERS` into two review tiers — any maintainer may review general code, while governance, legal, and CI/CD infrastructure files (`.github/`, `LICENSE`, `MAINTAINERS.md`, `SECURITY.md`, etc.) require sign-off from an NVIDIA maintainer. Kept identical across `fvdb-core`, `fvdb-reality-capture`, and `fvdb-examples` (#676 - @harrism).

## Version 0.4.2 - March 25, 2026

**Bug Fixes:**
- Build warning fix for GCC 14.

## Version 0.4.1 - March 25, 2026

**Bug Fixes:**
- Updated `nanovdb-editor` dependency to use version 0.0.23.

## Version 0.4.0 - March 12, 2026

*140 commits, 300+ files changed, 10 contributors.*

This release delivers major advances across the Gaussian splatting pipeline, sparse convolution, multi-GPU performance, and build/release infrastructure. fVDB now supports PyTorch 2.10 and CUDA 12.8/13.0, and ships its first formal release process with automated nightly builds.

**Highlights:**
- Gaussian splatting gains a new rasterize-from-world path that renders directly from 3D Gaussians, Unscented Transform projection for non-pinhole camera models, full MCMC splatting support, sparse rendering, per-pixel/per-tile masking, and a composable camera model that decouples kernels from camera internals. Numerous gradient correctness fixes harden the backward pass.
- Sparse convolution has been consolidated into a single GatherScatterDefault backend with full feature support including transposed convolution and arbitrary strides. A new PredGatherIGemm backend using CUTLASS/CuTe implicit-GEMM with TF32 tensor cores delivers significantly faster convolution on dense grids.
- A new multi-axis dispatch framework provides flexible kernel execution across multiple dimensions with typed views and for_each iteration.
- SampleGridTrilinear is roughly 2x faster via vectorized float4 loads and a fused stencil-plus-sample optimization. Morton and Hilbert space-filling curve ordering is now available for grid coordinates.
- Multi-GPU scaling is significantly improved through batched prefetching, device-centric synchronization, and radix sort optimizations. All tensor index accessors are now 64-bit, enabling larger datasets.
- A fully automated nightly wheel build and publish pipeline, a formal OneFlow release process with automation scripts, and GPU-validated publish workflows are all new in this release.

**Contributors:** @blackencino, @fwilliams, @harrism, @iYuqinL, @kmuseth, @matthewdcong, @phapalova, @areidmeyer, @swahtz, @zlalena

---

### Gaussian Splatting & Rendering


**New Features:**
- Added a new rasterization pathway that operates directly on **3D Gaussians** (#444 - @fwilliams).
- Added **Gaussian projection via the Unscented Transform**, providing an alternative to the EWA splatting approximation (#420 - @fwilliams).
- Added full **MCMC Gaussian Splatting** support, including relocation (#374) and add-noise (#377) kernels, Python bindings (#394), and tunable `min_opacity` (#396) and `k`/`t` (#402) parameters (@harrism, @fwilliams).
- Added end-to-end **sparse Gaussian rendering** with sparse rendering functions (#348) and sparse tile intersection (#401) (@fwilliams, @swahtz).
- Rasterization can now **render all contributing Gaussian IDs and weights** per pixel (#340 - @swahtz).
- Gaussian rasterization now supports **background colors** (#343 - @harrism).
- All Gaussian render methods now accept **`masks`** and **`backgrounds`** (#480 - @swahtz).
- The `evaluate_spherical_harmonics` function is now **exposed in the Python API** (#431 - @swahtz).
- Refactored the rendering pipeline around **composable camera operation classes** that encapsulate camera-space transform and projection, decoupling kernels from camera internals (#485 - @fwilliams), with a CameraIntrinsics constructor fix for host/device compatibility (#489 - @blackencino).

**Optimizations:**
- Switched the GaussianTileIntersection cumulative sum to use CUB for better performance (#427 - @swahtz).
- Optimized the `computeSparseInfo` path to reduce overhead in sparse rendering (#428 - @swahtz).
- Improved the contributing Gaussian ID kernels with shared-memory and loop optimizations (#429 - @swahtz).
- Optimized tile intersection for multi-GPU execution with better prefetching (#446 - @matthewdcong).
- Removed an unnecessary stream synchronization in GaussianTileIntersection (#370 - @harrism).
- `ProjectedGaussianSplats` opacities now use an efficient expand/view instead of per-element copy (#457 - @swahtz).

**Bug Fixes:**
- Fixed a shared memory alignment issue in the Gaussian rasterization kernel (#342 - @swahtz).
- Fixed inverted abs(gradient) logic in the backward rasterization pass that produced incorrect gradients (@harrism).
- Fixed NaN outputs in the top-contributing Gaussian IDs weights computation (#400 - @swahtz).
- Fixed camera data loading that could exceed blockDim when using many cameras (#345 - @swahtz).
- Fixed incorrect derivation of the number of cameras in packed rasterization mode (#414 - @swahtz).
- Fixed the chain rule for the log_scale gradient in the projection backward pass (#433 - @harrism).
- Fixed a race condition in the spherical harmonics backward pass when using multiple cameras or large batch sizes (#437 - @swahtz).
- Fixed the `dLossDQuat` quaternion gradient missing a warp-level reduction in the projection backward pass (#435, #533 - @swahtz, @matthewdcong).
- Fixed a multi-GPU race condition in the multibatch spherical harmonics backward pass (#484 - @matthewdcong).
- Fixed the ProjectionForward kernel double-initializing accessors, which caused correctness issues (#453 - @swahtz).
- Fixed a crash when loading GaussianPly files to a CPU device (#417 - @swahtz).
- Fixed handling of duplicate pixels in sparse pixel Gaussian rendering (#488 - @harrism).
- Fixed an incorrect datatype in the backward projection test (#486 - @matthewdcong).

---

### Sparse Convolution (Major)

- Consolidated all legacy sparse convolution backends into a single **GatherScatterDefault** backend with full feature support, including transposed convolution, arbitrary strides, and all float types (#473 - @blackencino).
- Added a new **PredGatherIGemm** sparse convolution backend using CUTLASS/CuTe implicit-GEMM with TF32 tensor cores, significantly faster than GatherScatterDefault for dense or near-dense grids (#508 - @blackencino).
- Fixed the default convolution behavior and added extensive correctness tests (#321 - @blackencino).
- Added gradient and backward pass tests to the convolution unit test suite (#358, #361 - @blackencino).
- Removed unused legacy sparse convolution backends (ImplicitGEMM, CUTLASS, LGGS, ME), deleting approximately 22,500 lines of code (#454 - @blackencino).
- Moved all op dispatch and precondition code into each op's C++ implementation files, making ops self-contained and reducing compile-time interconnectivity (#492 - @blackencino).

---

### Multi-Axis Dispatch Framework (New)

- Introduced a new **multi-axis dispatch framework** for flexible kernel execution across multiple dimensions (#418 - @blackencino).
- Extended the dispatch framework with `for_each` iteration, typed views, and tag canonicalization (#452 - @blackencino).
- The framework ships as a full C++ library under `src/dispatch/` with comprehensive tests and benchmarks.

---

### Grid Operations & Spatial Indexing

- Added **Morton and Hilbert** space-filling curve ordering for Grid and GridBatch ijk coordinates, with module-level standalone functions (#311, #316, #323 - @blackencino).
- `SampleGridTrilinear` now uses **vectorized float4 loads**, yielding roughly a 2x throughput improvement (#430 - @swahtz).
- `SampleGridTrilinear` received a second optimization pass using a fused stencil-plus-sample approach (#474 - @swahtz).
- Cleaned up the active grid coordinate generation code for clarity and consistency (#318 - @blackencino).

---

### JaggedTensor

- JaggedTensor reduce operators now support **bfloat16** (#501 - @swahtz).
- Fixed a binary search edge case in `JIdxForJOffsets` that returned incorrect indices when joffsets contained duplicate values (#325 - @iYuqinL).
- Fixed `from_*_and_list_ids` producing incorrect results with ldim=2 (#357 - @swahtz).
- Fixed concatenation errors in `JaggedTensor.jcat` (#352 - @blackencino).
- Reduced the number of blocking GPU-to-CPU copies in the `unbind*` methods, improving throughput (#363 - @swahtz).
- Fixed the single-element JaggedTensor constructor unconditionally initializing CUDA even for CPU tensors (#469 - @swahtz).

---

### Performance & Multi-GPU

- Optimized joffsets construction by using **pinned memory** to overlap CPU/GPU transfers (#403 - @matthewdcong).
- Significantly improved **multi-GPU scaling** through batched prefetching and sorting changes (#499 - @matthewdcong).
- Switched to device-centric synchronization for the forEach multi-GPU codepath (#440 - @matthewdcong).
- Fixed and improved radix sort synchronization across multiple rounds of improvements (#315, #409, #415 - @matthewdcong).
- Fused SSIM outputs now prefetch to avoid write page faults that degraded performance (#407 - @matthewdcong).
- MCMC kernels now support **PrivateUse1** for multi-GPU execution (#421 - @harrism).
- Switched from `torch.inverse` to `torch.linalg.inv_ex` to avoid an unnecessary device synchronization (#487 - @matthewdcong).
- All **32-bit tensor index accessors have been upgraded to 64-bit** across every op, enabling support for larger datasets (#505 - @harrism).

---

### PyTorch & CUDA Compatibility

- fVDB now builds and runs with **PyTorch 2.10** (#423, #521 - @matthewdcong, @swahtz).
- Added support for **CUDA 12.8 and 13.0** toolkits (#521 - @swahtz).
- Replaced the custom scaled dot-product attention implementation with PyTorch's native `torch.scaled_dot_product_attention` (#364 - @swahtz).
- Fixed the CCCL version check macro that could cause build failures with newer CUDA toolkits (#509 - @matthewdcong).
- Improved PyTorch build configuration time by streamlining CMake detection (#441 - @matthewdcong).

---

### NanoVDB

- Updated the bundled NanoVDB dependency to version 32.9.1 (#475, #483, #493 - @swahtz).
- Fixed voxel size and origin metadata not being preserved when serializing index grids (#490 - @swahtz).

---

### Neural Network Modules

- Fixed several bugs in SimpleUnet: NaN propagation from -inf values entering BatchNorm after max-pooling, incorrect ConvolutionPlan source/target grid assignments, and a crash on non-contiguous grad_output in the convolution backward pass (#496 - @swahtz).
- Added dedicated unit tests for all `fvdb.nn` modules to improve coverage and prevent regressions (#497 - @swahtz).

---

### Visualization / Viewer

- The viewer now supports displaying **multiple scenes** simultaneously with a scene-switching UI (#308 - @phapalova).
- Added viz bindings for `wait` and `add_image` to enable blocking display and image overlays (#332 - @phapalova).
- Fixed the viewer so it works correctly inside Jupyter notebooks (#350 - @zlalena).

---

### Build & Packaging

- Renamed the Python extension binary from `_Cpp` to `_fvdb_cpp` for clarity and to avoid naming conflicts (#317, #322 - @harrism, @blackencino).
- Improved build times with compilation speedups and added build tracing support (#443 - @blackencino).
- Fixed potential oversubscription when nvcc and cmake parallelism combined to exceed available cores (#351 - @swahtz).
- Added a `lineinfo` build option to include source-line debug info for GPU profiling (#367 - @harrism).
- Added a `getMaxSharedMemory` utility to centralize shared memory limit queries across kernels (#368 - @harrism).
- Added a `Version` class that provides structured version information at runtime (#507 - @swahtz).

---

### Nightly Builds & Release Infrastructure (New)

- Added a fully automated **nightly wheel build and publish pipeline** that builds across a matrix of Python, PyTorch, and CUDA versions and publishes to an S3 simple index (#477, #478 - @swahtz).
- Established a **formal release process** based on the OneFlow branching model, with `start-release.sh` and `finish-release.sh` automation scripts (#512, #525 - @harrism, @swahtz).
- The publish workflow now includes **GPU validation** with smoke tests and full unit tests on built wheels, an S3 staging index with automatic 30-day pruning, and support for release branch pushes triggering builds automatically.

---

### CI / DevOps

- Documentation-only PRs now auto-pass CI instead of showing a perpetual "waiting for status" indicator (#462 - @swahtz).
- Draft PRs now skip test runs entirely, saving compute resources (#339 - @swahtz).
- CI checkout references are pinned to immutable commit SHAs to prevent build/test skew between checkout and merge steps (#503 - @swahtz).
- Nightly workflows are now restricted to the upstream `openvdb/fvdb-core` repository and no longer run on forks (#319 - @harrism).
- Runner stop jobs are now skipped when the corresponding start job was skipped, avoiding spurious failures (#471, #472 - @harrism).
- Unit tests now only run for the matrix entry matching the `test_environment.yml` configuration (#531 - @harrism).

---

### Developer Tooling (New)

- Added **git worktree tools** (`fvdb-open`, `fvdb-close`, `fvdb-issue`) that make it easy to work on multiple branches simultaneously (#445 - @harrism).
- Added an **unanswered external issues reporter** with Slack output and a daily CI workflow to help the team stay on top of community questions (#510, #513 - @harrism).
- Added an `AGENTS.md` file providing persistent coding guidelines for AI agents working on the codebase (#455 - @harrism).

---

### Documentation

- Added and updated introductory, neural network, and convolution notebooks (#504 - @swahtz).
- Applied NVIDIA branding to the documentation site (#405 - @fwilliams).
- Added documentation for installing nightly builds from the S3 package index (#481 - @swahtz).
- Integrated Google Analytics into the documentation site for usage tracking (#312 - @fwilliams).
