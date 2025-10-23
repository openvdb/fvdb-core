# Installation

The `fvdb_core` Python package can be installed either using published packages with pip or built from
source.

## Prerequisites

fVDB is currently supported on the matrix of dependencies in the following table.

|    PyTorch    |   Python    |     CUDA     |
| --------------| ----------- | ------------ |
| 2.8.0 - 2.9.0 | 3.10 - 3.13 | 12.8 - 13.0  |

## Installation with pip

Currently, pip wheels are built with PyTorch 2.8 and CUDA 12.8 only. Versions for Python 3.10-3.13
are provided.

Install fvdb_core using the following pip command.

```
pip install fvdb_core==0.3.0+pt28.cu129 --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==2.8
```

## Installation from Source

Clone the *f*VDB source from [https://github.com/openvdb/fvdb-core](https://github.com/openvdb/fvdb-core),
then follow the instructions in the [README.md](https://github.com/openvdb/fvdb-core/README.md) under
"Building *f*VDB from Source". Environment and dependency management using Conda, Python virtual
environments, and Docker containers are covered.


