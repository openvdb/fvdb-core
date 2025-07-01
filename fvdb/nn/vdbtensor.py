# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass
from typing import Any

import torch
from fvdb.types import (
    JaggedTensorOrTensor,
    Vec3dBatch,
    Vec3dBatchOrScalar,
    Vec3i,
    is_JaggedTensorOrTensor,
)

from fvdb import GridBatch, JaggedTensor, SparseConvPackInfo, gridbatch_from_dense


@dataclass
class VDBTensor:
    """
    A VDBTensor is a thin wrapper around a GridBatch and its corresponding feature JaggedTensor, conceptually denoting a batch of
    sparse tensors along with its topology.
    It works as the input and output arguments of fvdb's neural network layers.
    One can simply construct a VDBTensor from a GridBatch and a JaggedTensor, or from a dense tensor using from_dense().
    """

    grid: GridBatch
    data: JaggedTensor

    # Only stores the kernel map that operates on this grid, reasons being:
    #   1) A usual network seldom re-uses computation for down-up-sampling. This saves memory.
    #   2) This keeps the implementation simple and the kmap transparent.
    kmap: SparseConvPackInfo | None = None

    def __post_init__(self):
        if not isinstance(self.grid, GridBatch):
            raise TypeError("grid should be of type GridBatch")
        if not isinstance(self.data, JaggedTensor):
            raise TypeError("data must be a JaggedTensor or a torch.Tensor")
        if self.grid.grid_count != len(self.data):
            raise ValueError("grid and feature should have the same batch size")
        if self.grid.total_voxels != self.data.jdata.size(0):
            raise ValueError("grid and feature should have the same total voxel count")
        if self.kmap is not None:
            if not isinstance(self.kmap, SparseConvPackInfo):
                raise TypeError("kmap should be of type SparseConvPackInfo")
            if not (
                self.is_same(self.kmap.source_grid)
                and self.is_same(self.kmap.target_grid)
                and self.kmap.stride == (1, 1, 1)
            ):
                raise ValueError("kmap should operate on the same grid as this tensor")

    def __getitem__(self, idx):
        return VDBTensor(self.grid[idx], self.data[idx])

    def __len__(self):
        return self.grid.grid_count

    def to_dense(self) -> torch.Tensor:
        # This would map grid.ijk.min() to dense_feature[0, 0, 0]
        return self.grid.write_to_dense(self.data)

    def clear_cache(self):
        self.kmap = None

    def is_same(self, other: Any):
        if isinstance(other, VDBTensor):
            return self.grid.address == other.grid.address and self.grid_count == other.grid_count
        else:
            # This will catch the case where other is a GridBatch or a GridBatchCpp
            return self.grid.has_same_address_and_grid_count(other)

    # -----------------------------
    # Arithmetic and math functions
    # -----------------------------
    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b)

    def __pow__(self, other):
        return self._binop(other, lambda a, b: a**b)

    def __neg__(self):
        return VDBTensor(self.grid, -self.data, self.kmap)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b)

    def __floordiv__(self, other):
        return self._binop(other, lambda a, b: a // b)

    def __mod__(self, other):
        return self._binop(other, lambda a, b: a % b)

    def __gt__(self, other):
        return self._binop(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._binop(other, lambda a, b: a < b)

    def __ge__(self, other):
        return self._binop(other, lambda a, b: a >= b)

    def __le__(self, other):
        return self._binop(other, lambda a, b: a <= b)

    def __eq__(self, other):
        return self._binop(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._binop(other, lambda a, b: a != b)

    def __iadd__(self, other):
        def inplace_add(a, b):
            a += b

        return self._binop_inplace(other, inplace_add)

    def __isub__(self, other):
        def inplace_sub(a, b):
            a -= b

        return self._binop_inplace(other, inplace_sub)

    def __imul__(self, other):
        def inplace_mul(a, b):
            a *= b

        return self._binop_inplace(other, inplace_mul)

    def __ipow__(self, other):
        def inplace_pow(a, b):
            a **= b

        return self._binop_inplace(other, inplace_pow)

    def __itruediv__(self, other):
        def inplace_truediv(a, b):
            a /= b

        return self._binop_inplace(other, inplace_truediv)

    def __ifloordiv__(self, other):
        def inplace_floordiv(a, b):
            a //= b

        return self._binop_inplace(other, inplace_floordiv)

    def __imod__(self, other):
        def inplace_mod(a, b):
            a %= b

        return self._binop_inplace(other, inplace_mod)

    def sqrt(self):
        return VDBTensor(self.grid, self.data.sqrt(), self.kmap)

    def abs(self):
        return VDBTensor(self.grid, self.data.abs(), self.kmap)

    def round(self):
        return VDBTensor(self.grid, self.data.round(), self.kmap)

    def floor(self):
        return VDBTensor(self.grid, self.data.floor(), self.kmap)

    def ceil(self):
        return VDBTensor(self.grid, self.data.ceil(), self.kmap)

    def sqrt_(self):
        self.data.sqrt_()
        return self

    def abs_(self):
        self.data.abs_()
        return self

    def round_(self):
        self.data.round_()
        return self

    def floor_(self):
        self.data.floor_()
        return self

    def ceil_(self):
        self.data.ceil_()
        return self

    def _binop(self, other, op):
        if isinstance(other, VDBTensor):
            return VDBTensor(self.grid, op(self.data, other.data), self.kmap)
        else:
            return VDBTensor(self.grid, op(self.data, other), self.kmap)

    def _binop_inplace(self, other, op):
        if isinstance(other, VDBTensor):
            op(self.data, other.data)
            return self
        else:
            op(self.data, other)
            return self

    # -----------------------
    # Interpolation functions
    # -----------------------

    def sample_bezier(self, points: JaggedTensorOrTensor) -> JaggedTensor:
        if not is_JaggedTensorOrTensor(points):
            raise TypeError(f"points should be a JaggedTensor or a torch.Tensor, but got {type(points)}")
        return self.grid.sample_bezier(points, self.data)  # type: ignore

    def sample_bezier_with_grad(self, points: JaggedTensorOrTensor) -> tuple[JaggedTensor, JaggedTensor]:
        if not is_JaggedTensorOrTensor(points):
            raise TypeError(f"points should be a JaggedTensor or a torch.Tensor, but got {type(points)}")
        return self.grid.sample_bezier_with_grad(points, self.data)  # type: ignore

    def sample_trilinear(self, points: JaggedTensorOrTensor) -> JaggedTensor:
        if not is_JaggedTensorOrTensor(points):
            raise TypeError(f"points should be a JaggedTensor or a torch.Tensor, but got {type(points)}")
        return self.grid.sample_trilinear(points, self.data)  # type: ignore

    def sample_trilinear_with_grad(self, points: JaggedTensorOrTensor) -> tuple[JaggedTensor, JaggedTensor]:
        if not is_JaggedTensorOrTensor(points):
            raise TypeError(f"points should be a JaggedTensor or a torch.Tensor, but got {type(points)}")
        return self.grid.sample_trilinear_with_grad(points, self.data)  # type: ignore

    def cpu(self):
        return VDBTensor(self.grid.to("cpu"), self.data.cpu(), self.kmap.cpu() if self.kmap is not None else None)

    def cuda(self):
        return VDBTensor(self.grid.to("cuda"), self.data.cuda(), self.kmap.cuda() if self.kmap is not None else None)

    def to(self, device_or_dtype: Any):
        return VDBTensor(
            self.grid.to(device_or_dtype),
            self.data.to(device_or_dtype),
            self.kmap.to(device_or_dtype) if self.kmap is not None else None,
        )

    def detach(self):
        return VDBTensor(self.grid, self.data.detach(), self.kmap)

    def type(self, arg0: torch.dtype):
        return VDBTensor(self.grid, self.data.type(arg0))

    def requires_grad_(self, required_grad):
        self.data.requires_grad_(required_grad)
        return self

    def clone(self):
        return VDBTensor(self.grid, self.data.clone(), self.kmap)

    @property
    def num_tensors(self):
        return self.data.num_tensors

    @property
    def is_cuda(self):
        return self.data.is_cuda

    @property
    def is_cpu(self):
        return self.data.is_cpu

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def jidx(self):
        return self.data.jidx

    @property
    def jlidx(self):
        return self.data.jlidx

    @property
    def joffsets(self):
        return self.data.joffsets

    @property
    def jdata(self):
        return self.data.jdata

    @property
    def rshape(self):
        return self.data.rshape

    @property
    def lshape(self):
        return self.data.lshape

    @property
    def ldim(self):
        return self.data.ldim

    @property
    def eshape(self):
        return self.data.eshape

    @property
    def edim(self):
        return self.data.edim

    @property
    def requires_grad(self):
        return self.data.requires_grad

    @property
    def cum_voxels(self) -> torch.LongTensor:
        cv = self.grid.cum_voxels
        if not isinstance(cv, torch.LongTensor):
            raise TypeError(f"cum_voxels should be a torch.LongTensor, but got {type(cv)}")
        return cv

    @property
    def grid_count(self) -> int:
        return self.grid.grid_count

    @property
    def ijk(self) -> JaggedTensor:
        return self.grid.ijk

    @property
    def num_voxels(self) -> torch.LongTensor:
        nv = self.grid.num_voxels
        if not isinstance(nv, torch.LongTensor):
            raise TypeError(f"num_voxels should be a torch.LongTensor, but got {type(nv)}")
        return nv

    @property
    def origins(self) -> torch.FloatTensor:
        o = self.grid.origins
        if not isinstance(o, torch.FloatTensor):
            raise TypeError(f"origins should be a torch.FloatTensor, but got {type(o)}")
        return o

    @property
    def total_voxels(self) -> int:
        return self.grid.total_voxels

    @property
    def voxel_sizes(self) -> torch.FloatTensor:
        vs = self.grid.voxel_sizes
        if not isinstance(vs, torch.FloatTensor):
            raise TypeError(f"voxel_sizes should be a torch.FloatTensor, but got {type(vs)}")
        return vs

    @property
    def total_leaf_nodes(self) -> int:
        return self.grid.total_leaf_nodes

    @property
    def num_leaf_nodes(self) -> torch.LongTensor:
        nl = self.grid.num_leaf_nodes
        if not isinstance(nl, torch.LongTensor):
            raise TypeError(f"num_leaf_nodes should be a torch.LongTensor, but got {type(nl)}")
        return nl

    @property
    def grid_to_world_matrices(self) -> torch.FloatTensor:
        gtwm = self.grid.grid_to_world_matrices
        if not isinstance(gtwm, torch.FloatTensor):
            raise TypeError(f"grid_to_world_matrices should be a torch.FloatTensor, but got {type(gtwm)}")
        return gtwm

    @property
    def world_to_grid_matrices(self) -> torch.FloatTensor:
        wtg = self.grid.world_to_grid_matrices
        if not isinstance(wtg, torch.FloatTensor):
            raise TypeError(f"world_to_grid_matrices should be a torch.FloatTensor, but got {type(wtg)}")
        return wtg

    @property
    def bbox(self) -> torch.IntTensor:
        b = self.grid.bbox
        if not isinstance(b, torch.IntTensor):
            raise TypeError(f"bbox should be a torch.IntTensor, but got {type(b)}")
        return b

    @property
    def dual_bbox(self) -> torch.IntTensor:
        db = self.grid.dual_bbox
        if not isinstance(db, torch.IntTensor):
            raise TypeError(f"dual_bbox should be a torch.IntTensor, but got {type(db)}")
        return db

    @property
    def total_bbox(self) -> torch.IntTensor:
        tb = self.grid.total_bbox
        if not isinstance(tb, torch.IntTensor):
            raise TypeError(f"total_bbox should be a torch.IntTensor, but got {type(tb)}")
        return tb


def vdbtensor_from_dense(
    dense_data: torch.Tensor,
    ijk_min: Vec3i | None = None,
    voxel_sizes: Vec3dBatchOrScalar | None = None,
    origins: Vec3dBatch | None = None,
) -> VDBTensor:
    if origins is None:
        origins = [0.0] * 3
    if voxel_sizes is None:
        voxel_sizes = [1.0] * 3
    if ijk_min is None:
        ijk_min = [0, 0, 0]
    assert ijk_min is not None
    grid = gridbatch_from_dense(
        dense_data.size(0),
        dense_data.size()[1:4],
        ijk_min=ijk_min,
        voxel_sizes=voxel_sizes,
        origins=origins,
        device=dense_data.device,
    )
    # Note: this would map dense_feature[0, 0, 0] to grid[ijk_min]
    data = grid.read_from_dense(dense_data.contiguous(), dense_origins=ijk_min)
    return VDBTensor(grid, data)
