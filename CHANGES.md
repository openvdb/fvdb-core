fVDB Version History
====================

Version 0.5.0 - In Development

Version 0.4.0 - March 12, 2026

*140 commits, 300+ files changed, 10 contributors.*

This release delivers major advances across the Gaussian splatting
pipeline, sparse convolution, multi-GPU performance, and build/release
infrastructure. fVDB now supports PyTorch 2.10 and CUDA 12.8/13.0, and
ships its first formal release process with automated nightly builds.

**Highlights:**
- Gaussian splatting gains a new rasterize-from-world path that
  renders directly from 3D Gaussians, Unscented Transform projection
  for non-pinhole camera models, full MCMC splatting support, sparse
  rendering, per-pixel/per-tile masking, and a composable camera model
  that decouples kernels from camera internals. Numerous gradient
  correctness fixes harden the backward pass.
- Sparse convolution has been consolidated into a single
  GatherScatterDefault backend with full feature support including
  transposed convolution and arbitrary strides. A new PredGatherIGemm
  backend using CUTLASS/CuTe implicit-GEMM with TF32 tensor cores
  delivers significantly faster convolution on dense grids.
- A new multi-axis dispatch framework provides flexible kernel
  execution across multiple dimensions with typed views and for_each
  iteration.
- SampleGridTrilinear is roughly 2x faster via vectorized float4 loads
  and a fused stencil-plus-sample optimization. Morton and Hilbert
  space-filling curve ordering is now available for grid coordinates.
- Multi-GPU scaling is significantly improved through batched
  prefetching, device-centric synchronization, and radix sort
  optimizations. All tensor index accessors are now 64-bit, enabling
  larger datasets.
- A fully automated nightly wheel build and publish pipeline, a formal
  OneFlow release process with automation scripts, and GPU-validated
  publish workflows are all new in this release.

**Contributors:** @blackencino, @fwilliams, @harrism, @iYuqinL,
@kmuseth, @matthewdcong, @phapalova, @areidmeyer, @swahtz, @zlalena

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
