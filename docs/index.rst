Welcome to ƒVDB!
=================

fVDB is a Python library of data structures and algorithms for building high-performance and large-domain
spatial applications using `NanoVDB <https://dl.acm.org/doi/abs/10.1145/3450623.3464653>`_ on the GPU
in `PyTorch <https://pytorch.org/>`_.
Applications of fVDB include 3D deep learning, computer graphics/vision, robotics, and scientific computing.

.. raw:: html

  <video autoplay loop controls muted width="90%" style="display: block; margin: 0 auto;">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/fvdb_intro_480p.mp4" type="video/mp4" />
  </video>

|

fVDB aims to be production ready with a focus on robustness, usability, and extensibility.
It is designed to be easily integrated into existing pipelines and workflows, and to support a
wide range of use cases and applications. To this end, fVDB has a minimal set of dependencies and
is open source under the Apache 2.0 license as part of the `The Academy Software Foundation's
OpenVDB project <https://www.openvdb.org>`_.  Contributions and feedback from the community are welcome
to fVDB's `GitHub repository <https://github.com/openvdb/fvdb-core>`_.


Features
--------

fVDB provides the following key features:

-   A sparse volumetric grid data structure optimized for GPU memory efficiency and performance.
-   A highly optimized Gaussian splat data structure for representing radiance fields on the GPU.
-   A jagged tensor data structure for efficient representation of sparse, non-uniform data on the GPU.
-   A suite of GPU-accelerated algorithms for volumetric data manipulation, ray tracing, and volume rendering.
-   A state of the art visualizer capable of streaming massive volumetric datasets to a web browser or Jupyter notebook.
-   Modular neural network components for building 3D deep learning models that scale to large input sizes.
-   Seamless integration with PyTorch for easy use in deep learning workflows.

The videos below show fVDB being used for large-scale 3D reconstruction, simulation, and interactive visualization.

.. raw:: html

   <p style="text-align: center; font-weight: bold; font-style: italic; text-decoration: underline; font-size: medium; text-decoration-skip-ink: none; margin-bottom: 0.5em;">
   fVDB being used to reconstruct radiance fields and TSDF volumes from images and points</p>
  <video autoplay loop controls muted width="90%" style="display: block; margin: 0 auto;">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/spot_airport_480p.mp4" type="video/mp4" />
  </video>

   <br>

   <p style="text-align: center; font-weight: bold; font-style: italic; text-decoration: underline; font-size: medium; text-decoration-skip-ink: none; margin-bottom: 0.5em;">
   fVDB being used to visualize large scale volumetric data in a web browser</p>
  <video autoplay loop controls muted width="90%" style="display: block; margin: 0 auto;">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/large_recon_480p.mp4" type="video/mp4" />
  </video>

|

About fVDB
--------------

fVDB was first developed by the `NVIDIA High-Fidelity Physics Research Group <https://research.nvidia.com/labs/prl/>`_
within the `NVIDIA Spatial Intelligence Lab <https://research.nvidia.com/labs/sil/>`_, and continues to be
developed with the OpenVDB community to suit the growing needs for a robust framework for
spatial intelligence research and applications.

.. toctree::
   :caption: Introduction
   :hidden:

   self

.. toctree::
   :maxdepth: 1
   :caption: Basic Concepts

.. toctree::
   :maxdepth: 1
   :caption: API References

   api/jagged_tensor
   api/convolution_plan
   api/sparse_grids
   api/gaussian_splatting
   api/viz
   api/enums
   api/nn
   api/utils

.. raw:: html

   <hr>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
