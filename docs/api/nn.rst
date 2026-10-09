Neural Network Layers and Blocks
===================================

Call modules as ``module(...)`` to retain PyTorch hooks. The ``forward`` entries
below describe their arguments and outputs.

.. autoclass:: fvdb.nn.MaxPool
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.AvgPool
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.UpsamplingNearest
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.Prune
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SparseConv3d
   :members:
   :inherited-members: Module
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SparseConvTranspose3d
   :members:
   :inherited-members: Module
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.BatchNorm
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.GroupNorm
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SyncBatchNorm
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.DropPath
   :members:
   :exclude-members: extra_repr


U-Net Architecture Blocks
---------------------------

.. autoclass:: fvdb.nn.SimpleUNet
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SimpleUNetBasicBlock
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SimpleUNetBottleneck
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SimpleUNetConvBlock
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SimpleUNetDown
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SimpleUNetDownUp
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SimpleUNetPad
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SimpleUNetUnpad
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SimpleUNetUp
   :members:
   :exclude-members: extra_repr


Point Transformer V3
--------------------

Serialized-attention layers and network following Pointcept's Point Transformer V3.
Patch and window attention need PyTorch 2.11 or newer and an SM80+ GPU.

.. autoclass:: fvdb.nn.PointTransformerV3
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.PointTransformerV3Block
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SerializedAttention
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.ConditionalPositionEncoding
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SerializedPooling
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.SerializedUnpooling
   :members:
   :exclude-members: extra_repr

.. autoclass:: fvdb.nn.PointTransformerV3Embedding
   :members:
   :exclude-members: extra_repr
