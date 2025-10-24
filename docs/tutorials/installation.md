# Installation

The `fvdb_core` Python package can be installed either using published packages with pip or built
from source.

## Prerequisites

fVDB is currently supported on the matrix of dependencies in the following table.

|    PyTorch    |   Python    |     CUDA     |
| --------------| ----------- | ------------ |
| 2.8.0 - 2.9.0 | 3.10 - 3.13 | 12.8 - 13.0  |

 - Linux is the only platform currently supported (Ubuntu >= 22.04 recommended).
 - A CUDA-capable GPU with Ampere architecture or newer (i.e. compute capability >=8.0) is
   recommended to run the CUDA-accelerated operations in ƒVDB. A GPU with compute capabililty >=7.0
   (Volta architecture) is the minimum requirement but some operations and data types are not
   supported.

## Installation with pip

Currently, pip wheels are built with PyTorch 2.8 and CUDA 12.8 only. Versions for Python 3.10-3.13
are provided.

Install fvdb_core using the following pip command.

```
pip install fvdb_core==0.3.0+pt28.cu129 --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==2.8
```

## Installation from Source

Clone the [`fvdb-core` source](https://github.com/openvdb/fvdb-core), then follow the instructions in
the [README.md](https://github.com/openvdb/fvdb-core/blob/main/README.md#building-fvdb-from-source)
under "Building *f*VDB from Source". Environment and dependency management using Conda, Python
virtual environments, and Docker containers are covered.


