Neural Network Layers and Blocks
===================================

Call modules as ``module(...)`` to retain PyTorch hooks. The ``forward`` entries
below describe their arguments and outputs.

.. autoclass:: fvdb.nn.MaxPool
   :members: forward

.. autoclass:: fvdb.nn.AvgPool
   :members: forward

.. autoclass:: fvdb.nn.UpsamplingNearest
   :members: forward

.. autoclass:: fvdb.nn.Prune
   :members: forward

.. autoclass:: fvdb.nn.SparseConv3d
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SparseConvTranspose3d
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.BatchNorm
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.GroupNorm
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SyncBatchNorm
   :members: forward, reset_parameters


U-Net Architecture Blocks
---------------------------

.. autoclass:: fvdb.nn.SimpleUNet
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SimpleUNetBasicBlock
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SimpleUNetBottleneck
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SimpleUNetConvBlock
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SimpleUNetDown
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SimpleUNetDownUp
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SimpleUNetPad
   :members: forward, reset_parameters, create_padded_grid

.. autoclass:: fvdb.nn.SimpleUNetUnpad
   :members: forward, reset_parameters

.. autoclass:: fvdb.nn.SimpleUNetUp
   :members: forward, reset_parameters
