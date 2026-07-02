Installing fVDB
================================================================

fVDB depends on `PyTorch <https://pytorch.org/>`_, and requires a CUDA-capable GPU. Below are the
supported software and hardware configurations.

Software Requirements
------------------------

The following is a compatibility matrix of the versions of software compatible with each minor release of fVDB.  These are the versions of software fVDB was built and tested against that are officially supported:

+--------------+------------------+-----------------+-----------------+----------------+------------------------------------------+
| fVDB Version | Operating System | PyTorch Version | Python Version  | CUDA Version   | Vulkan Version (only for visualization)  |
+--------------+------------------+-----------------+-----------------+----------------+------------------------------------------+
| 0.5          | Linux Only       | 2.11.0          | 3.10 - 3.14     | 12.8, 13.0     | 1.3.275.0                                |
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

Installation from conda-forge
------------------------------
To install ``fvdb-core`` in a conda environment, run the following command to install the latest released version of ``fvdb-core`` from `conda-forge <https://anaconda.org/conda-forge/fvdb-core>`_:

.. code-block:: bash

    conda install --channel conda-forge fvdb-core

Installation from pre-built wheels
-------------------------------------
To install ``fvdb-core`` using pip, run the appropriate pip install command for your Pytorch/CUDA versions. These commands will install
the correct version of ``fvdb-core`` if it is not already installed.


PyTorch 2.11.0 + CUDA 13.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

    pip install fvdb-core==\ |fvdb_core_version_pt211_cu130| --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu130_tag|

PyTorch 2.11.0 + CUDA 12.8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

    pip install fvdb-core==\ |fvdb_core_version_pt211_cu128| --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu128_tag|

.. note::
   Visualization and viewer features additionally require the ``nanovdb_editor`` Python package. Install it using the optional 'viewer' dependencies, by adding ``[viewer]`` to the ``fvdb-core`` package name, for example: ``pip install fvdb-core[viewer]==…``.


Installation from nightly builds
-------------------------------------

Nightly wheels are built from the latest ``main`` branch and published daily.
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
    selects the highest local version, which today is the CUDA 13.0 build. To
    target a different build (for example, CUDA 12.8) or pin a specific date
    for reproducibility, use one of the explicit commands below.

PyTorch 2.11.0 + CUDA 13.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

    pip install fvdb-core==\ |fvdb_core_nightly_base|\ .dev20260428+pt\ |torch_short|\ .\ |cu130_tag| --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple-nightly" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu130_tag|

PyTorch 2.11.0 + CUDA 12.8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

    pip install fvdb-core==\ |fvdb_core_nightly_base|\ .dev20260428+pt\ |torch_short|\ .\ |cu128_tag| --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple-nightly" torch==\ |torch_full_version| --extra-index-url https://download.pytorch.org/whl/|cu128_tag|

To list all available nightly versions:

.. code-block:: bash

    pip index versions fvdb-core --index-url="https://d36m13axqqhiit.cloudfront.net/simple-nightly" --pre

.. note::

    Replace ``20260428`` with the desired nightly date. Nightly builds are retained for 30 days.


Installation from source
-----------------------------

.. note::

    For more complete instructions including setting up a build environment and obtaining the
    necessary dependencies, see the fVDB `README <https://github.com/openvdb/fvdb-core/blob/main/README.md>`_.


Clone the `fvdb-core repository <https://github.com/openvdb/fvdb-core>`_.

.. code-block:: bash

   git clone git@github.com:openvdb/fvdb-core.git

Next build and install the fVDB library

.. code-block:: bash

   pushd fvdb-core
   ./build.sh install verbose
   popd
