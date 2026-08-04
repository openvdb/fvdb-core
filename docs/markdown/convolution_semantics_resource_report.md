# Convolution topology resource report

This is the Section 13.6 release measurement for issue #668. The reproducible
harness is [`measure_topology_resources.py`](../../src/benchmarks/convolution/measure_topology_resources.py).
It uses the installed extension, starts a fresh subprocess for every CPU case,
and fails if a required direct or count-then-fill invariant is not observed.

## Method and environment

The measurement was run on 2026-08-03 from commit
`622c475c1b9be198e3e01ac5fdf4c34f89a5f7c6`, before the final release-gate
reinstall, with:

```console
source /home/chorvath/miniforge3/etc/profile.d/conda.sh
conda activate fvdb
python src/benchmarks/convolution/measure_topology_resources.py \
  --execution --output /tmp/convolution_topology_resources.json
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

All rows below assert `valid_emission_count == N`, `used_direct_projection`,
and `peak_requested_bytes == 16*N`. Thus K=S=2 retains its proved direct
floor-coarsening path; K=S=3 and 4 use the phase-aware direct projection. The
requested staging is independent of kernel volume.

| Device | K=S | N | Candidate rows | Build ms | Requested staging MiB | Observed peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 2 | 4,096 | 32,768 | 1.063 | 0.062 | 0.781 HWM |
| CPU | 3 | 4,096 | 110,592 | 1.055 | 0.062 | 0.500 HWM |
| CPU | 4 | 4,096 | 262,144 | 0.855 | 0.062 | 0.531 HWM |
| CPU | 2 | 32,768 | 262,144 | 1.574 | 0.500 | 0.500 HWM |
| CPU | 3 | 32,768 | 884,736 | 1.810 | 0.500 | 0.250 HWM |
| CPU | 4 | 32,768 | 2,097,152 | 1.361 | 0.500 | 0.500 HWM |
| CPU | 2 | 110,592 | 884,736 | 3.361 | 1.688 | 0.281 HWM |
| CPU | 3 | 110,592 | 2,985,984 | 2.985 | 1.688 | 0.250 HWM |
| CPU | 4 | 110,592 | 7,077,888 | 1.903 | 1.688 | 0.250 HWM |
| CUDA | 2 | 4,096 | 32,768 | 10.597 | 0.062 | 0.354 allocated / 0 reserved |
| CUDA | 3 | 4,096 | 110,592 | 170.319 | 0.062 | 0.355 allocated / 0 reserved |
| CUDA | 4 | 4,096 | 262,144 | 164.606 | 0.062 | 0.355 allocated / 0 reserved |
| CUDA | 2 | 32,768 | 262,144 | 10.808 | 0.500 | 0.792 allocated / 0 reserved |
| CUDA | 3 | 32,768 | 884,736 | 168.384 | 0.500 | 1.002 allocated / 0 reserved |
| CUDA | 4 | 32,768 | 2,097,152 | 174.600 | 0.500 | 1.002 allocated / 0 reserved |
| CUDA | 2 | 110,592 | 884,736 | 10.386 | 1.688 | 2.109 allocated / 0 reserved |
| CUDA | 3 | 110,592 | 2,985,984 | 159.218 | 1.688 | 3.377 allocated / 2.000 reserved |
| CUDA | 4 | 110,592 | 7,077,888 | 204.543 | 1.688 | 3.377 allocated / 2.000 reserved |

## Other required topology cases

These are the largest geometric-sweep input (`N=110,592`). `M` is the exact
forward emission count; the count-fill cases assert `!used_direct_projection`,
`emission_requested_bytes == 16*M`, and `M < candidate rows`.

| Device | Case | K / S | Candidate rows | M or generated output | Requested peak MiB | Build ms | Observed peak MiB | Final grid bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CPU | issue #668 direct | (4,1,1) / (4,1,1) | 442,368 | M=110,592 | 1.688 | 3.562 | 0.500 HWM | 312,032 |
| CUDA | issue #668 direct | (4,1,1) / (4,1,1) | 442,368 | M=110,592 | 1.688 | 179.129 | 3.377 allocated / 2.000 reserved | 312,032 |
| CPU | K<S count-fill | (3,3,3) / (4,4,4) | 2,985,984 | M=46,656 | 1.556 | 14.081 | 0.250 HWM | 305,888 |
| CUDA | K<S count-fill | (3,3,3) / (4,4,4) | 2,985,984 | M=46,656 | 1.556 | 177.631 | 2.956 allocated / 2.000 reserved | 305,888 |
| CPU | K>S count-fill | (4,4,4) / (3,3,3) | 7,077,888 | M=262,144 | 4.844 | 36.196 | 0.281 HWM | 307,712 |
| CUDA | K>S count-fill | (4,4,4) / (3,3,3) | 7,077,888 | M=262,144 | 4.844 | 196.122 | 8.002 allocated / 2.000 reserved | 307,712 |
| CPU | generated transpose | (4,4,4) / (4,4,4) | 7,077,888 | output=7,077,888 | n/a | 88.014 | 38.000 HWM | 4,578,400 |
| CUDA | generated transpose | (4,4,4) / (4,4,4) | 7,077,888 | output=7,077,888 | n/a | 180.815 | 218.014 allocated / 220.000 reserved | 4,578,400 |

The raw JSON includes every case at all three geometric sizes, including the
generated-transpose sweep: CPU build time is 7.519, 31.489, and 88.014 ms for
`N=4,096`, `32,768`, and `110,592`; CUDA is 174.674, 171.061, and 180.815 ms.
The measurement harness rejects cases over 8,000,000 exact candidate rows by
default so the sweep cannot accidentally consume the machine. The library does
not use that fixed benchmark limit: it checks the exact
`16*N*kernel_volume` staging size with overflow-safe arithmetic before
allocation, then lets PyTorch's allocator decide whether the construction fits.
If allocation fails, fVDB preserves the `OutOfMemoryError` type and reports the
exact coordinate-staging requirement with actionable context. A predictive
capacity gate is intentionally avoided because inactive split cache is reusable
by the allocator even though it cannot be released to the CUDA driver.

## Lazy plan-coverage follow-up

On 2026-08-03, after the original recorded sweep commit, the post-review
worktree was installed and the CUDA plan path was measured again on the same
machine with a prebuilt full-support target for a dense `128^3` source and
`K=3,S=1`. The rulebook has 56,623,104 pairs between 2,097,152 input rows and
2,197,000 output rows. Three synchronized plan constructions took 62.395,
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
| K=S=2 | 0.504 |
| K=S=3 | 1.210 |
| K=S=4 | 2.912 |
| issue #668 `(4,1,1)/(4,1,1)` | 0.341 |
| K<S `(3,3,3)/(4,4,4)` | 1.105 |
| K>S `(4,4,4)/(3,3,3)` | 2.567 |

## Checked ten-million-voxel estimate

For `N=10,000,000`, uniform `K=S=4` has `4^3=64` old candidate rows per input.
The former coordinate, batch-index, and mask lower bound is checked as

```text
10,000,000 * 64 * (12 + 4 + 1) = 10,880,000,000 bytes = 10.13 GiB.
```

The direct path reports exactly one 16-byte coordinate/batch emission per
input, or `160,000,000` bytes = 152.59 MiB, before final-grid construction.
This is `O(N)` rather than `O(N*K_volume)` and is asserted by the harness for
K=S=2, 3, and 4. The direct result is requested staging accounting, not a claim
that it includes every allocator or final-grid byte.

## Limitations and interpretation

- Timings include synchronization and topology construction, but are single
  samples after worker startup rather than statistical throughput measurements.
- CUDA reserved bytes depend on allocator history and are secondary diagnostics.
- CPU HWM deltas may undercount as described above; the native requested-staging
  fields are the exact allocation-contract evidence for forward topology.
- Generated transpose legitimately emits all taps and therefore scales with its
  full output population. The harness's 8,000,000-row cap is an intentional
  measurement guard; the library instead validates the exact staging size and
  contextualizes an allocator-authoritative out-of-memory failure. This avoids
  false rejection when a request fits reusable split cache.
- The original sweep tables were collected against the installed implementation
  at the commit recorded above. The lazy-coverage follow-up was collected from
  the later installed post-review worktree described in its section. Re-run the
  measurements after compiler/runtime changes.
