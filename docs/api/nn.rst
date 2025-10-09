fvdb.nn
=======

`fvdb.nn` is a collection of neural network layers to build sparse neural networks.

Pooling and Upsampling
-----------------------

.. autoclass:: fvdb.nn.MaxPool
.. autoclass:: fvdb.nn.AvgPool
.. autoclass:: fvdb.nn.UpsamplingNearest

Convolution Layers
------------------

.. autoclass:: fvdb.nn.SparseConv3d
.. autoclass:: fvdb.nn.SparseConvTranspose3d

Normalization Layers
--------------------

.. autoclass:: fvdb.nn.BatchNorm
.. autoclass:: fvdb.nn.GroupNorm
.. autoclass:: fvdb.nn.SyncBatchNorm

Activation Functions
--------------------

.. autoclass:: fvdb.nn.CELU
.. autoclass:: fvdb.nn.ELU
.. autoclass:: fvdb.nn.GELU
.. autoclass:: fvdb.nn.LeakyReLU
.. autoclass:: fvdb.nn.ReLU
.. autoclass:: fvdb.nn.SELU
.. autoclass:: fvdb.nn.Sigmoid
.. autoclass:: fvdb.nn.SiLU
.. autoclass:: fvdb.nn.Tanh

Other Layers
------------

.. autoclass:: fvdb.nn.Dropout
.. autoclass:: fvdb.nn.InjectFromGrid
.. autoclass:: fvdb.nn.Linear
.. autoclass:: fvdb.nn.Sequential

U-Net Architecture
------------------

.. autoclass:: fvdb.nn.SimpleUNet
.. autoclass:: fvdb.nn.SimpleUNetBasicBlock
.. autoclass:: fvdb.nn.SimpleUNetBottleneck
.. autoclass:: fvdb.nn.SimpleUNetConvBlock
.. autoclass:: fvdb.nn.SimpleUNetDown
.. autoclass:: fvdb.nn.SimpleUNetDownUp
.. autoclass:: fvdb.nn.SimpleUNetPad
.. autoclass:: fvdb.nn.SimpleUNetUnpad
.. autoclass:: fvdb.nn.SimpleUNetUp
