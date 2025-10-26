# *ƒ*VDB

fVDB is a Python library of data structures and algorithms for building high-performance and large-domain
spatial applications using [NanoVDB](https://dl.acm.org/doi/abs/10.1145/3450623.3464653) on the GPU
in [PyTorch](https://pytorch.org/).
Applications of fVDB include 3D deep learning, computer graphics/vision, robotics, and scientific computing.


<p align="center">
  <img src="docs/imgs/fvdb_teaser.jpg" style="width: 90%;" alt="fVDB Teaser">
</p>



fVDB was first developed by the [NVIDIA High-Fidelity Physics Research Group](https://research.nvidia.com/labs/prl/)
within the [NVIDIA Spatial Intelligence Lab](https://research.nvidia.com/labs/sil/), and continues to be
developed with the OpenVDB community to suit the growing needs for a robust framework for
spatial intelligence research and applications.

[The paper](https://research.nvidia.com/labs/prl/publication/williams2024fvdb/) is available for more details, kindly consider [citing it in your work](#references) if you find it useful.

## Learning to Use *f*VDB

After [installing *f*VDB](#installing-fvdb), we recommend starting with our [documentation](https://fvdb.ai/).

Beyond the [documentation](https://fvdb.ai/), the walk-through [notebooks](notebooks) in this repository
can provide an illustrated introduction to the main concepts in *f*VDB.


## Installing *f*VDB

The `fvdb-core` Python package can be installed either using published packages with pip or built
from source.

For the most up-to-date information on installing *f*VDB's pip packages, please see the
[installation documentation](https://fvdb.ai/installation.html).

## Building *f*VDB from Source

If the [pre-built packages](https://fvdb.ai/installation.html) do not meet your needs, you can build *f*VDB from source in this repository.

### Environment Management

ƒVDB is a Python library implemented as a C++ PyTorch extension. We provide three paths to
constructing reliable environments for building and running ƒVDB. These are separate options not
intended to be used together (however with modification you can of course use, for example, a conda
or pip environment inside a docker container).

1. **RECOMMENDED** [conda](#option-1-setting-up-a-conda-environment-recommended)
2. Using [docker](#option-2-setting-up-a-docker-container)
3. Python virtual environment. [venv](#option-3-setting-up-a-python-virtual-environment)

`conda` tends to be more flexible since reconfiguring toolchains and modules to suit your larger
project can be dynamic, but at the same time this can be a more brittle experience compared to using
a virtualized `docker` container.  Using `conda` is generally recommended for development and
testing, while using `docker` is recommended for CI/CD and deployment.

#### **OPTION 1** Conda Environment (Recommended)

*f*VDB can be used with any Conda distribution installed on your system. Below is an installation guide using
[miniforge](https://github.com/conda-forge/miniforge). You can skip steps 1-3 if you already have a Conda installation.

1. Download and Run Install Script. Copy the command below to download and run the [miniforge install script](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos--linux):

```shell
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

2. Follow the prompts to customize Conda and run the install. Note, we recommend saying yes to enable `conda-init`.

3. Start Conda. Open a new terminal window, which should now show Conda initialized to the `(base)` environment.

4. Create the `fvdb` conda environment. Run the following command from the directory containing this README:

```shell
conda env create -f env/dev_environment.yml
```

5. Activate the *f*VDB environment:

```shell
conda activate fvdb
```

##### Other available environments
* `fvdb_build`: Use `env/build_environment.yml` for a minimum set of dependencies needed just to build/package *f*VDB (note this environment won't have all the runtime dependencies needed to `import fvdb`).
* `fvdb_test`: Use `env/test_environment.yml` for a runtime environment which has only the packages required to run the unit tests after building ƒVDB. This is the environment used by the CI pipeline to run the tests after building ƒVDB in the `fvdb_build` environment.
* `fvdb_learn`: Use `env/learn_environment.yml` for additional runtime requirements and packages needed to run the [notebooks](notebooks) or [examples](examples) and view their visualizations.

---

#### **OPTION 2** Docker Container

Running a Docker container ensures that you have a consistent environment for building and running ƒVDB. Start by installing Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Our provided [`Dockerfile`](Dockerfile) constructs a container that pre-installs the dependencies needed to build and run ƒVDB.

1. In the fvdb-core directory, build the Docker image:
```shell
docker build -t fvdb-devel .
```

2. When you are ready to build ƒVDB, run the following command within the docker container.  `TORCH_CUDA_ARCH_LIST` specifies which CUDA architectures to build for.
```shell
docker run -it --mount type=bind,src="$(pwd)",target=/workspace fvdb-devel bash
cd /workspace;
pip install -r env/build_requirements.txt
TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0;10.0;12.0+PTX" \
./build.sh install -v
```

In order to extract an artifact from the container such as the Python wheel, query the container ID using `docker ps` and copy the artifact using `docker cp`.

---

#### **OPTION 3** Python Virtual Environment

Using a Python virtual environment enables you to use your system provided compiler and CUDA toolkit. This can be especially useful if you are using ƒVDB in conjunction with other Python packages, especially packages that have been built from source.

1. Start by installing GCC, the CUDA Toolkit, and cuDNN.

2. Then, create a Python virtual environment, install the requisite dependencies, and build:

```shell
python -m venv fvdb
source fvdb/bin/activate
pip install -r env/build_requirements.txt
TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0;10.0;12.0+PTX" ./build.sh install -v
```

Note: adjust the TORCH_CUDA_ARCH_LIST to suit your needs. If you are building just to run on a single machine, including only the present GPU architecture(s) reduces build time.

---

### Building *f*VDB

#### Tips for Building *f*VDB

  - **:warning:** Compilation can be very memory-consuming. As part of our build script, we set the `CMAKE_BUILD_PARALLEL_LEVEL` environment variable to control compilation job parallelism with a value that we find works well for most machines (allowing for one job every 2.5GB of memory) but this can be overridden by setting the `CMAKE_BUILD_PARALLEL_LEVEL` environment variable to a different value.

  - To save time and trouble on repeated clean builds, configure your `CPM_SOURCE_CACHE`. Add the following to your shell configuration (e.g. `.bashrc`)

    ```shell
    export CPM_SOURCE_CACHE=$HOME/.cache/CPM
    ```

    If this is not set, CMake Package Manager (CPM) will cache in the fVDB build directory. Keeping the cache outside of the build directory allows build-time dependencies
    to be reused across fvdb clean-build cycles and saves build time. [See the CPM documentation for more detail](https://github.com/cpm-cmake/CPM.cmake?tab=readme-ov-file#cpm_source_cache)

#### Build Commands

You can either perform an install:
```shell
./build.sh
```

or if you would like to build a packaged wheel for installing in other environments, you can run the following command:
```shell
./build.sh wheel
```

The build script automatically detects the CUDA architectures to build for based on the available GPUs on the system. You can override this behavior by setting the `--cuda-arch-list` option.

```shell
./build.sh --cuda-arch-list=8.0;8.6+PTX
```

#### Build Modifiers

The build script supports the following build modifiers:

- `gtests`: Enable building the gtest C++ unit tests.
- `benchmarks`: Enable building the benchmarks.
- `editor_skip`: Skip building the nanovdb_editor dependency.
- `editor_force`: Force rebuild of the nanovdb_editor dependency.
- `debug`: Build in debug mode with full debug symbols and no optimizations.
- `strip_symbols`: Strip symbols from the build (will be ignored if debug is enabled).
- `verbose`: Enable verbose build output for pip and CMake.

### Running Tests

#### C++ Tests

To run the gtest C++ unit tests

```shell
./build.sh ctest
```

#### Python Tests

To run the pytests

```shell
cd tests
pytest unit
```

### Building Documentation

To build the documentation, simply run:
```shell
sphinx-build ./docs -a -E build/sphinx
# View the docs
open build/sphinx/index.html
# View docs as served
cd build/sphinx
python -m http.server
# Open localhost:8000 in browser
```

### Setting up Intellisense with clangd in Visual Studio Code

Please see the guide [`Clangd for Intellisense in fVDB`](docs/markdown/clangd.md)


## Code Structure
The main source code for fVDB lives in the [src](src) directory. There are several important files here:
* `src/python/Bindings.cpp` exposes functionality directly to Python. It is mainly a wrapper around the core classes such as `fvdb::GridBatch` and `fvdb::JaggedTensor`.
* `src/GridBatch.h` contains the implementation of `fvdb::GridBatch` which is the core data structure on which fVDB is built. A `GridBatch` acts as a map between `(i, j, k)` voxel coordinates and offsets in linear memory. This mapping can be used to perform a host of operations. The methods in this class are mostly lightweight wrappers around a set of CPU and CUDA *kernels*. The function prototypes for these kernels are defined in `src/detail/ops/*.h`.
* `src/detail/ops/*.h` contains the function prototypes for the main kernels used by fVDB. Host and device kernel implementations are provided in the `src/detail/ops/*.cu` source files.
* `src/detail/autograd` contains C++ implementations of PyTorch autograd functions for differentiable operations.  `#include <detail/autograd/Autograd.h>` includes all of the functions in this directory.
* `src/detail/utils/nanovdb` contains a number of utilities which make it easier to use NanoVDB.


## References

Please consider citing the following technical paper, presented at ACM SIGGRAPH 2024, when adopting *f*VDB in your project:

```bibtex
@article{williams2024fvdb,
  author = {Williams, Francis and Huang, Jiahui and Swartz, Jonathan and Klar, Gergely and Thakkar, Vijay and Cong, Matthew and Ren, Xuanchi and Li, Ruilong and Fuji-Tsang, Clement and Fidler, Sanja and Sifakis, Eftychios and Museth, Ken},
  title = {fVDB : A Deep-Learning Framework for Sparse, Large Scale, and High Performance Spatial Intelligence},
  year = {2024},
  issue_date = {July 2024},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {43},
  number = {4},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3658226},
  doi = {10.1145/3658226},
  journal = {ACM Trans. Graph.},
  month = jul,
  articleno = {133},
  numpages = {15},
}
```

## Contact

For questions or feedback, please use the [GitHub Issues](https://github.com/openvdb/fvdb-core/issues) for this repository.
