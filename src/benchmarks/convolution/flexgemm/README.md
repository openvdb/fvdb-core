# Sparse convolution benchmarks

Scripts comparing fVDB `ConvolutionPlan` backends with each other and with
[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM)'s Triton masked implicit GEMM, mostly on
FlexGEMM's submanifold training benchmark (voxelized sphere shells).

| Script | Purpose |
|---|---|
| `bench_train.py` | L-layer forward + backward step; `--topology per-iter` rebuilds grid and plan each step as FlexGEMM's README does, `--topology fixed` builds them once |
| `bench_layer.py` | Stem and head layers (C_in or C_out of 1-4) forward + backward, which masked_igemm runs zero-padded |
| `bench_strided.py` | One strided conv (default 2x2x2, stride 2) forward + backward against FlexGEMM's `sparse_conv3d`, on the configs of FlexGEMM's `tests/spconv/bwd.py`; also checks that both outputs match |
| `profile_split.py` | One layer split into grid build, plan build, forward, backward; `--kernels` prints per-kernel tables |
| `check.py` | Forward, input-gradient and weight-gradient agreement with FlexGEMM |
| `bench_kernel_sizes.py` | One layer forward, forward + backward and plan build for kernel sizes 3, 5, 7: `masked_igemm` against the default backend and `pred_gather_igemm` (no FlexGEMM needed) |
| `plan_memory.py` | GPU memory a plan holds, and its build peak, for `masked_igemm` and the default backend |
| `common.py` | Workloads, timing, backend adapters |

## Setup

FlexGEMM builds a CUDA extension against the active torch:

```
E=$CONDA_PREFIX
git clone https://github.com/JeffreyXiang/FlexGEMM && cd FlexGEMM
CUDA_HOME=$E PATH=$E/bin:$PATH $E/bin/python -m pip install -e . --no-build-isolation --no-deps
```

On Blackwell, Triton needs `TRITON_PTXAS_BLACKWELL_PATH=$(which ptxas)`.

## Usage

```
python bench_train.py --mode fp16 --topology fixed --backends default,masked_igemm,flex_gemm --json out.json
python bench_strided.py --mode fp16 --topology fixed --json strided.json
python profile_split.py --mode fp16 --backends masked_igemm,flex_gemm
python check.py --backends masked_igemm --dtypes fp16
python bench_kernel_sizes.py --json kernel_sizes.json
python plan_memory.py --kind shell --res 512 --kernel-size 7 --stride 1
```

Results from runs on an RTX PRO 6000 Blackwell (SM120), CUDA 13.0, PyTorch 2.13, Triton 3.7.1 and
FlexGEMM 6dd94a8 are kept on the
[`conv-masked-igemm-benchmark-archive`](https://github.com/swahtz/fvdb-core/tree/conv-masked-igemm-benchmark-archive/src/benchmarks/convolution/flexgemm/results)
branch, with the tile-tuning and prototype scripts used during development.
