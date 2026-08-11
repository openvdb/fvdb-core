Installing fVDB
================================================================

fVDB depends on `PyTorch <https://pytorch.org/>`_, and is accelerated for CUDA-capable GPUs. Below are the
supported software and hardware configurations.

Software Requirements
------------------------

The following is a matrix of the versions of software that we `test and distribute pre-built wheels for <#notes-on-testing-and-distribution>`_ with each minor release of fVDB.

+--------------+------------------+-----------------+-----------------+----------------+------------------------------------------+
| fVDB Version | Operating System | PyTorch Version | Python Version  | CUDA Version   | Vulkan Version (only for visualization)  |
+--------------+------------------+-----------------+-----------------+----------------+------------------------------------------+
| 0.6          | Linux Only       | 2.13.0          | 3.10 - 3.15     | 13.0, 13.2     | 1.3.275.0                                |
+--------------+------------------+-----------------+-----------------+----------------+------------------------------------------+
| 0.5          | Linux Only       | 2.11.0          | 3.10 - 3.14     | 13.0, 13.2     | 1.3.275.0                                |
+--------------+------------------+-----------------+-----------------+----------------+------------------------------------------+
| 0.4          | Linux Only       | 2.10.0          | 3.10 - 3.13     | 12.8, 13.0     | 1.3.275.0                                |
+--------------+------------------+-----------------+-----------------+----------------+------------------------------------------+
| 0.3          | Linux Only       | 2.8.0           | 3.10 - 3.13     | 12.8           | 1.3.275.0                                |
+--------------+------------------+-----------------+-----------------+----------------+------------------------------------------+

Driver and Hardware Requirements
-----------------------------------

The following table specifies the minimum NVIDIA driver versions and GPU architectures needed to run fVDB:

+------------------+----------------+------------------+---------------------+
| Operating System | Driver Version | GPU Architecture | Compute Capability  |
+------------------+----------------+------------------+---------------------+
| Linux Only       | 550.0 or later | Ampere or later  | 8.0 or greater      |
+------------------+----------------+------------------+---------------------+

While fVDB operators run on the CPU and CUDA-accelerated GPUs and is currently only built and tested on Linux, the `NanoVDB <https://www.openvdb.org/documentation/doxygen/NanoVDB_FAQ.html>`_ library which underlies fVDB is hardware and operating system agnostic.  fVDB is a community project and we welcome any contributors and collaborators interested in working on extending fVDB to other hardware platforms and operating systems; please reach out by opening an issue on the `fvdb-core repository <https://github.com/openvdb/fvdb-core/issues>`_.

Installation from conda-forge
------------------------------
To install ``fvdb-core`` in a conda environment, run the following command to install the latest released version of ``fvdb-core`` from `conda-forge <https://anaconda.org/conda-forge/fvdb-core>`_:

.. code-block:: bash

    conda install --channel conda-forge fvdb-core

Installation from pre-built wheels
-------------------------------------
To install ``fvdb-core`` using pip, run the appropriate pip install command for your Pytorch/CUDA versions. These commands will install
the correct version of ``fvdb-core`` if it is not already installed.


PyTorch 2.13.0 + CUDA 13.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

    pip install fvdb-core==\ |fvdb_core_version_pt213_cu132| --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu132_tag|

PyTorch 2.13.0 + CUDA 13.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::
    pip install fvdb-core==\ |fvdb_core_version_pt213_cu130| --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu130_tag|

.. note::
   Visualization and viewer features additionally require the ``nanovdb_editor`` Python package. Install it using the optional 'viewer' dependencies, by adding ``[viewer]`` to the ``fvdb-core`` package name, for example: ``pip install fvdb-core[viewer]==…``.


Installation from nightly builds
-------------------------------------

Wheels are built from the latest ``main`` branch and published on a nightly basis.
Each nightly version is anchored to the next upcoming release recorded in
``pyproject.toml`` (currently |fvdb_core_nightly_base|) and carries a date
stamp plus PyTorch/CUDA build identifiers, for example
|fvdb_core_nightly_base|\ .dev20260428+pt\ |torch_short|\ .\ |cu130_tag|.

Under PEP 440 ordering, each nightly sorts between the in-development version
on ``main`` and the corresponding final release, so passing ``--pre`` together
with the nightly index URL will track the latest nightly until that release
ships, then prefer the final release once it is tagged.

Latest nightly (any supported PyTorch/CUDA build)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

    pip install --pre fvdb-core --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple-nightly" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu130_tag|

.. note::

    The nightly index hosts wheels for every supported PyTorch/CUDA combination
    in a single project listing. Without an explicit local-version pin, ``pip``
    selects the highest local version, which today is the CUDA 13.2 build. To
    target a different build (for example, CUDA 13.0) or pin a specific date
    for reproducibility, use one of the explicit commands below.

PyTorch 2.13.0 + CUDA 13.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

    pip install fvdb-core==\ |fvdb_core_nightly_base|\ .dev20260428+pt\ |torch_short|\ .\ |cu132_tag| --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple-nightly" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu132_tag|

PyTorch 2.13.0 + CUDA 13.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

    pip install fvdb-core==\ |fvdb_core_nightly_base|\ .dev20260428+pt\ |torch_short|\ .\ |cu130_tag| --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple-nightly" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu130_tag|

To list all available nightly versions:

.. code-block:: bash

    pip index versions fvdb-core --index-url="https://d36m13axqqhiit.cloudfront.net/simple-nightly" --pre

.. note::

    Replace ``20260428`` with the desired nightly date. Nightly builds are retained for 30 days.


Build and install a custom wheel from source
---------------------------------------------

If you need a production wheel for a specific Python/PyTorch/CUDA combination
(or a custom set of CUDA architectures) that is not distribute in any of our
pre-built wheels, the repository provides a script that automates this process.
This script reproduces the official release build process inside a Docker
container — this is the same script our publish workflows use. It builds from
your local checkout (including any uncommitted changes) and copies the finished wheel to ``./dist/``.

The only host requirements are Docker (with BuildKit) and ``python3``; no GPU or CUDA toolkit is needed on the host unless you use ``--cuda-arch-list native``, which detects your GPU's compute capability via ``nvidia-smi``.

.. note::

    This script is to help produce a one-off installable wheel.  If you plan on doing active
    development on fVDB, we recommend seeing the `Development Process section below <#development-process>`_.

Clone the `fvdb-core repository <https://github.com/openvdb/fvdb-core>`_.

.. code-block:: bash

   git clone git@github.com:openvdb/fvdb-core.git
   cd fvdb-core


Build a wheel with the default versions (recorded in ``.github/versions.json``):

.. code-block:: bash

   ./docker/build_wheel.py

Or pick specific versions — for example Python 3.11, PyTorch 2.11.0, with CUDA 13.0, targeting
only the GPU architecture present on this machine:

.. code-block:: bash

   ./docker/build_wheel.py --python 3.11 --torch 2.11.0 --cuda 13.0 --cuda-arch-list native

The set of valid PyTorch/CUDA/Python combinations is determined by which wheels
PyTorch publishes on `download.pytorch.org <https://download.pytorch.org/whl/>`_
(the human-readable summary is on the `PyTorch get-started page
<https://pytorch.org/get-started/locally/>`_). Before starting the Docker build,
the script consults that package index and exits early with the list of
available alternatives if no PyTorch wheel exists for the requested
combination — for example, requesting ``--torch 2.13.0 --cuda 12.8`` will
report the PyTorch versions actually published for CUDA 12.8. If the index
cannot be reached (for example, when building offline), the check is skipped
with a warning and an invalid combination will instead fail during the build's
dependency-installation step.

Run ``./docker/build_wheel.py --help`` for all options. The resulting wheel
carries a ``+pt<torch>.cu<cuda>`` local version suffix (for example
``+pt211.cu130``) and must be installed alongside the matching PyTorch build:

.. parsed-literal::

   pip install torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu130_tag|
   pip install dist/fvdb_core-\*.whl


Development Process
---------------------

For more information about the development process, including instructions for setting up a build environment and obtaining the
necessary dependencies we recommend for development, see the fVDB `README <https://github.com/openvdb/fvdb-core/blob/main/README.md>`_.


Notes on Testing, Compatibility, and Distribution
--------------------------------------------------
An fvdb-core minor release is tested against the current stable minor version of PyTorch at the time of release and the latest two minor versions of CUDA compatible with that release from `PyTorch's release compatibility matrix <https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix>`_. fvdb-core minor releases will be distributed as a set of wheels built across a matrix of that same current, stable version of PyTorch; the latest two CUDA versions supported by the stable PyTorch version; and all Python minor versions supported by the stable PyTorch version\ [*]_. Any fvdb-core patch releases made for that minor version will maintain the same compatibility as the minor release.

Additionally, we test fvdb-core (on a weekly schedule) with the oldest PyTorch version that supports the lowest CUDA version which our 'stable PyTorch's latest two CUDA versions' policy above would imply.  For example, by PyTorch's matrix, for 2.13, the latest two supported CUDA versions are 13.2 and 13.0 and the lowest PyTorch version to have support for CUDA 13.0 is PyTorch 2.9.

Inside of the version ranges of our testing regime, the maintainers will review submitted fixes and work to fix reported issues. However, for compatibility issues outside that range, the maintainers will endeavor to assist but may not be able to resolve issues outside this scope. Generally, fixes and new features are targeted for the current in-development minor release and compatibility range.


.. [*] Builds of other combinations can be built `with this process <#build-and-install-a-custom-wheel-from-source>`_.
