# Convolution topology resource report

This is the Section 13.6 release measurement for issue #668. The reproducible
harness is [`measure_topology_resources.py`](../../src/benchmarks/convolution/measure_topology_resources.py).
It uses the installed extension, starts a fresh subprocess for every CPU case,
and fails if a required direct or count-then-fill invariant is not observed.

## Method and environment

The measurement was rerun on 2026-08-04 from the installed merge result combining
issue-668 head `5a6fe5e0e5b436bd1655df9bb2c3dae70152e99c` with upstream `main` at
`df517d97f5bb912fc614e93b80f5b4ac81e6be49`, with:

```console
source /home/chorvath/miniforge3/etc/profile.d/conda.sh
conda activate fvdb
python src/benchmarks/convolution/measure_topology_resources.py \
  --execution --output /tmp/convolution_topology_resources_post_merge.json
```

| Component | Value |
|---|---|
| fVDB | 0.6.0, installed from `.../envs/fvdb/lib/python3.12/site-packages/fvdb` |
| Python / PyTorch / CUDA | 3.12.13 / 2.10.0 / 13.0 |
| CPU | AMD Ryzen Threadripper PRO 5975WX, 32 logical CPUs |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, capability 12.0, 95.59 GiB |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2, x86_64 |

Each input is a dense one-batch cube with side 16, 32, or 48 (`N=4,096`,
`32,768`, or `110,592`). `candidate rows` is `N * kernel_volume`; it is an
upper bound for generated transpose and the old masked-forward candidate count.
Forward resource fields come from the test-visible native accounting. CUDA
peaks are `torch.cuda.max_memory_{allocated,reserved}` above an idle baseline;
reserved is allocator diagnostic rather than live tensor memory. CPU is the
delta in `ru_maxrss` in a fresh worker after grid setup. It is an observed
subprocess high-water delta and can undercount allocations that fit below the
import/setup high-water mark.

## Direct K=S geometric sweep

All rows below assert `valid_emission_count == N` and `used_direct_projection`.
CUDA `K=S=2` now uses upstream's NanoVDB leaf-mask coarsening and therefore
reports zero explicit coordinate staging. CPU `K=S=2` retains the proved direct
floor-coarsening realization, while `K=S=3` and `4` use the phase-aware direct
projection and request exactly one 16-byte coordinate/batch row per input.

| Device | K=S | N | Candidate rows | Build ms | Requested staging MiB | Observed peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 2 | 4,096 | 32,768 | 0.931 | 0.062 | 0.535 HWM |
| CPU | 3 | 4,096 | 110,592 | 0.995 | 0.062 | 0.535 HWM |
| CPU | 4 | 4,096 | 262,144 | 0.978 | 0.062 | 0.535 HWM |
| CPU | 2 | 32,768 | 262,144 | 1.621 | 0.500 | 0.535 HWM |
| CPU | 3 | 32,768 | 884,736 | 1.361 | 0.500 | 0.535 HWM |
| CPU | 4 | 32,768 | 2,097,152 | 1.454 | 0.500 | 0.535 HWM |
| CPU | 2 | 110,592 | 884,736 | 3.203 | 1.688 | 0.285 HWM |
| CPU | 3 | 110,592 | 2,985,984 | 2.520 | 1.688 | 0.262 HWM |
| CPU | 4 | 110,592 | 7,077,888 | 1.878 | 1.688 | 0.285 HWM |
| CUDA | 2 | 4,096 | 32,768 | 9.996 | 0.000 | 0.294 allocated / 0 reserved |
| CUDA | 3 | 4,096 | 110,592 | 208.527 | 0.062 | 0.354 allocated / 0 reserved |
| CUDA | 4 | 4,096 | 262,144 | 185.199 | 0.062 | 0.354 allocated / 0 reserved |
| CUDA | 2 | 32,768 | 262,144 | 8.647 | 0.000 | 0.294 allocated / 0 reserved |
| CUDA | 3 | 32,768 | 884,736 | 179.855 | 0.500 | 1.002 allocated / 0 reserved |
| CUDA | 4 | 32,768 | 2,097,152 | 165.898 | 0.500 | 1.002 allocated / 0 reserved |
| CUDA | 2 | 110,592 | 884,736 | 10.410 | 0.000 | 0.296 allocated / 0 reserved |
| CUDA | 3 | 110,592 | 2,985,984 | 200.624 | 1.688 | 3.377 allocated / 2.000 reserved |
| CUDA | 4 | 110,592 | 7,077,888 | 193.042 | 1.688 | 3.377 allocated / 2.000 reserved |

## Other required topology cases

These are the largest geometric-sweep input (`N=110,592`). `M` is the exact
forward emission count; the count-fill cases assert `!used_direct_projection`,
`emission_requested_bytes == 16*M`, and `M < candidate rows`.

| Device | Case | K / S | Candidate rows | M or generated output | Requested peak MiB | Build ms | Observed peak MiB | Final grid bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CPU | issue #668 direct | (4,1,1) / (4,1,1) | 442,368 | M=110,592 | 1.688 | 3.698 | 0.535 HWM | 312,032 |
| CUDA | issue #668 direct | (4,1,1) / (4,1,1) | 442,368 | M=110,592 | 1.688 | 167.803 | 3.377 allocated / 2.000 reserved | 312,032 |
| CPU | K<S count-fill | (3,3,3) / (4,4,4) | 2,985,984 | M=46,656 | 1.556 | 14.563 | 0.285 HWM | 305,888 |
| CUDA | K<S count-fill | (3,3,3) / (4,4,4) | 2,985,984 | M=46,656 | 1.556 | 171.722 | 2.956 allocated / 2.000 reserved | 305,888 |
| CPU | K>S count-fill | (4,4,4) / (3,3,3) | 7,077,888 | M=262,144 | 4.844 | 36.938 | 0.285 HWM | 307,712 |
| CUDA | K>S count-fill | (4,4,4) / (3,3,3) | 7,077,888 | M=262,144 | 4.844 | 178.369 | 8.002 allocated / 2.000 reserved | 307,712 |
| CPU | generated transpose | (4,4,4) / (4,4,4) | 7,077,888 | output=7,077,888 | n/a | 60.063 | 38.285 HWM | 4,578,400 |
| CUDA | generated transpose | (4,4,4) / (4,4,4) | 7,077,888 | output=7,077,888 | n/a | 186.728 | 218.014 allocated / 220.000 reserved | 4,578,400 |

The raw JSON includes every case at all three geometric sizes, including the
generated-transpose sweep: CPU build time is 6.764, 25.028, and 60.063 ms for
`N=4,096`, `32,768`, and `110,592`; CUDA is 169.869, 172.912, and 186.728 ms.
The measurement harness rejects cases over 8,000,000 exact candidate rows by
default so the sweep cannot accidentally consume the machine. The library's
coordinate fallback does not use that fixed benchmark limit: it checks the exact
`16*N*kernel_volume` staging size with overflow-safe arithmetic before
allocation, then lets PyTorch's allocator decide whether the construction fits.
If allocation fails, fVDB preserves the `OutOfMemoryError` type and reports the
exact coordinate-staging requirement with actionable context. A predictive
capacity gate is intentionally avoided because inactive split cache is reusable
by the allocator even though it cannot be released to the CUDA driver. The
generated-transpose rows above deliberately use shifted `K=S=4`, which requires
the coordinate fallback; unshifted `K=S={1,2}`, uniform stride-one windows, and
`K=3,S=2` use NanoVDB leaf-mask topology operations without candidate staging.

## Lazy plan-coverage follow-up

On 2026-08-03, the post-review worktree was installed and the CUDA plan path was
measured on the same machine with a prebuilt full-support target for a dense
`128^3` source and `K=3,S=1`. The rulebook has 56,623,104 pairs between
2,097,152 input rows and 2,197,000 output rows. Three synchronized plan constructions took 62.395,
63.996, and 62.483 ms (62.483 ms median), with 432.001 MiB peak allocation above
the source/target baseline for the retained int32 rulebook indices.

Full degree histograms are no longer part of that construction path. Their
first explicit access took 72.327 ms and 32.762 MiB peak allocation above the
completed plan, exactly the scale of the two int64 degree vectors rather than
either 56.6-million-element index cast. Cached access took 0.007 ms. Constructing
the exact plan transpose took 0.296 ms, and accessing its swapped cached report
took 0.013 ms without scanning the reversed rulebook. These are synchronized
single-run diagnostic timings except for the three-sample construction median;
they characterize allocation and latency boundaries rather than throughput.

## Execution sanity timing

The following are absolute median execution times at `N=110,592` with 8 input
and 8 output channels, five synchronized repetitions, on CUDA. They establish
that the corrected topology paths execute; there is no pre-change execution
baseline in this release measurement, so these are not a regression claim.

| Geometry | Median execution ms |
|---|---:|
| K=S=2 | 0.500 |
| K=S=3 | 1.332 |
| K=S=4 | 2.140 |
| issue #668 `(4,1,1)/(4,1,1)` | 0.299 |
| K<S `(3,3,3)/(4,4,4)` | 1.061 |
| K>S `(4,4,4)/(3,3,3)` | 2.862 |

## Checked ten-million-voxel estimate

For `N=10,000,000`, uniform `K=S=4` has `4^3=64` old candidate rows per input.
The former coordinate, batch-index, and mask lower bound is checked as

```text
10,000,000 * 64 * (12 + 4 + 1) = 10,880,000,000 bytes = 10.13 GiB.
```

The direct path reports exactly one 16-byte coordinate/batch emission per
input, or `160,000,000` bytes = 152.59 MiB, before final-grid construction.
This is `O(N)` rather than `O(N*K_volume)` and is asserted by the harness for
K=S=3 and 4. Uniform CUDA `K=S=2` is also `O(N)`, but uses NanoVDB leaf-mask
coarsening and reports zero explicit coordinate staging. These requested-staging
figures do not include every allocator or final-grid byte.

## Limitations and interpretation

- Timings include synchronization and topology construction, but are single
  samples after worker startup rather than statistical throughput measurements.
- CUDA reserved bytes depend on allocator history and are secondary diagnostics.
- CPU HWM deltas may undercount as described above; the native requested-staging
  fields are the exact allocation-contract evidence for forward topology.
- Coordinate-fallback transpose legitimately emits all taps and therefore scales
  with its full output population. The harness's 8,000,000-row cap is an
  intentional measurement guard; the library instead validates the exact
  fallback staging size and contextualizes an allocator-authoritative
  out-of-memory failure. This avoids false rejection when a request fits reusable
  split cache.
- The sweep tables were collected against the installed merged implementation
  identified above. The lazy-coverage follow-up predates the merge but measures a
  separate plan-construction path unaffected by the topology-builder integration.
