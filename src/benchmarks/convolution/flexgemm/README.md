# fVDB vs FlexGEMM sparse convolution benchmarks

Scripts comparing fVDB `ConvolutionPlan` backends with
[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM)'s Triton masked implicit GEMM on
FlexGEMM's submanifold training benchmark (voxelized sphere shells).

| Script | Purpose |
|---|---|
| `bench_train.py` | L-layer forward + backward step; `--topology per-iter` rebuilds grid and plan each step as FlexGEMM's README does, `--topology fixed` builds them once |
| `bench_layer.py` | Stem and head layers (C_in or C_out of 1-4) forward + backward, which masked_igemm runs zero-padded |
| `bench_strided.py` | One strided conv (default 2x2x2, stride 2) forward + backward against FlexGEMM's `sparse_conv3d`, on the configs of FlexGEMM's `tests/spconv/bwd.py`; also checks that both outputs match |
| `profile_split.py` | One layer split into grid build, plan build, forward, backward; `--kernels` prints per-kernel tables |
| `check.py` | Forward, input-gradient and weight-gradient agreement with FlexGEMM |
| `proto_fvdb_topology.py` | fVDB neighbor map driving FlexGEMM's Triton kernels under different row orderings |
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
```

`results/` holds JSON from runs on an RTX PRO 6000 Blackwell (SM120), CUDA 13.0, PyTorch 2.13,
Triton 3.7.1, FlexGEMM 6dd94a8. `baseline_2026-10-13/` holds the original comparison report.
