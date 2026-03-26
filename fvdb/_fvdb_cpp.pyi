# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import typing
from enum import Enum
from typing import ClassVar, Optional, overload

import torch

from .types import (
    DeviceIdentifier,
    ListOfListsOfTensors,
    ListOfTensors,
    LShapeSpec,
    NumericMaxRank1,
    NumericMaxRank2,
    NumericMaxRank3,
    RShapeSpec,
    Vec3iOrScalar,
)

class version:
    nanovdb: str
    cuda: str
    torch: str

class GatherScatterDefaultTopology:
    """Precomputed compacted CSR topology for default gather-scatter sparse convolution."""

    @property
    def gather_indices(self) -> torch.Tensor: ...
    @property
    def scatter_indices(self) -> torch.Tensor: ...
    @property
    def offsets(self) -> torch.Tensor: ...
    @property
    def feature_total_voxels(self) -> int: ...
    @property
    def output_total_voxels(self) -> int: ...
    @property
    def kernel_volume(self) -> int: ...
    @property
    def total_pairs(self) -> int: ...
    @property
    def kernel_size(self) -> list[int]: ...
    @property
    def stride(self) -> list[int]: ...
    @property
    def is_transposed(self) -> bool: ...

# Forward topology + conv
def gs_build_topology(
    feature_grid: GridBatchData,
    output_grid: GridBatchData,
    kernel_size: Vec3iOrScalar,
    stride: Vec3iOrScalar,
) -> GatherScatterDefaultTopology: ...
def gs_conv(
    features: torch.Tensor,
    weights: torch.Tensor,
    topology: GatherScatterDefaultTopology,
) -> torch.Tensor: ...
def gs_conv_backward(
    grad_output: torch.Tensor,
    features: torch.Tensor,
    weights: torch.Tensor,
    topology: GatherScatterDefaultTopology,
) -> tuple[torch.Tensor, torch.Tensor]: ...

# Transposed topology + conv
def gs_build_transpose_topology(
    feature_grid: GridBatchData,
    output_grid: GridBatchData,
    kernel_size: Vec3iOrScalar,
    stride: Vec3iOrScalar,
) -> GatherScatterDefaultTopology: ...
def gs_conv_transpose(
    features: torch.Tensor,
    weights: torch.Tensor,
    topology: GatherScatterDefaultTopology,
) -> torch.Tensor: ...
def gs_conv_transpose_backward(
    grad_output: torch.Tensor,
    features: torch.Tensor,
    weights: torch.Tensor,
    topology: GatherScatterDefaultTopology,
) -> tuple[torch.Tensor, torch.Tensor]: ...

# PredGatherIGemm convolution (SM80 CUTLASS IGEMM)
def pred_gather_igemm_conv(
    features: torch.Tensor,
    weights: torch.Tensor,
    feature_grid: GridBatchData,
    output_grid: GridBatchData,
    kernel_size: int,
    stride: int,
) -> torch.Tensor: ...

class GaussianSplat3d:
    class ProjectionType(Enum):
        PERSPECTIVE = ...
        ORTHOGRAPHIC = ...

    log_scales: torch.Tensor
    logit_opacities: torch.Tensor
    means: torch.Tensor
    quats: torch.Tensor
    requires_grad: bool
    sh0: torch.Tensor
    shN: torch.Tensor
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
        accumulate_mean_2d_gradients: bool = ...,
        accumulate_max_2d_radii: bool = ...,
        detach: bool = ...,
    ) -> None: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @staticmethod
    def cat(
        splats: "list[GaussianSplat3d]",
        accumulate_mean_2d_gradients: bool = False,
        accumulate_max_2d_radii: bool = False,
        detach: bool = False,
    ) -> "GaussianSplat3d": ...
    def to(self, device: torch.device, dtype: torch.dtype) -> "GaussianSplat3d": ...
    def detach(self) -> "GaussianSplat3d": ...
    def detach_in_place(self) -> None: ...
    def index_select(self, indices: torch.Tensor) -> "GaussianSplat3d": ...
    def mask_select(self, mask: torch.Tensor) -> "GaussianSplat3d": ...
    def slice_select(self, begin: int, end: int, step: int) -> "GaussianSplat3d": ...
    def index_set(self, indices: torch.Tensor, value: "GaussianSplat3d") -> None: ...
    def mask_set(self, mask: torch.Tensor, value: "GaussianSplat3d") -> None: ...
    def slice_set(self, begin: int, end: int, step: int, value: "GaussianSplat3d") -> None: ...
    @property
    def sh_degree(self) -> int: ...
    @property
    def accumulate_mean_2d_gradients(self) -> bool: ...
    @accumulate_mean_2d_gradients.setter
    def accumulate_mean_2d_gradients(self, value: bool) -> None: ...
    @property
    def accumulate_max_2d_radii(self) -> bool: ...
    @accumulate_max_2d_radii.setter
    def accumulate_max_2d_radii(self, value: bool) -> None: ...
    @staticmethod
    def from_state_dict(state_dict: dict[str, torch.Tensor]) -> GaussianSplat3d: ...
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None: ...
    def project_gaussians_for_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
    ) -> ProjectedGaussianSplats: ...
    def project_gaussians_for_images(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        sh_degree_to_use: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
    ) -> ProjectedGaussianSplats: ...
    def project_gaussians_for_images_and_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        sh_degree_to_use: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
    ) -> ProjectedGaussianSplats: ...
    def render_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
        backgrounds: Optional[torch.Tensor] = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def sparse_render_depths(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...
    def render_from_projected_gaussians(
        self,
        projected_gaussians: ProjectedGaussianSplats,
        crop_width: int = ...,
        crop_height: int = ...,
        crop_origin_w: int = ...,
        crop_origin_h: int = ...,
        tile_size: int = ...,
        backgrounds: Optional[torch.Tensor] = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def render_images(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        sh_degree_to_use: int = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
        backgrounds: Optional[torch.Tensor] = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def render_images_from_world(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: "DistortionModel" = ...,
        distortion_coeffs: Optional[torch.Tensor] = ...,
        sh_degree_to_use: int = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
        backgrounds: Optional[torch.Tensor] = ...,
        masks: Optional[torch.Tensor] = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def sparse_render_images(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        sh_degree_to_use: int = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...
    def render_images_and_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        sh_degree_to_use: int = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
        backgrounds: Optional[torch.Tensor] = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def sparse_render_images_and_depths(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        sh_degree_to_use: int = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...
    def render_num_contributing_gaussians(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def sparse_render_num_contributing_gaussians(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...
    def render_contributing_gaussian_ids(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
        top_k_contributors: int = ...,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...
    def sparse_render_contributing_gaussian_ids(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type: ProjectionType = ...,
        tile_size: int = ...,
        min_radius_2d: float = ...,
        eps_2d: float = ...,
        antialias: bool = ...,
        top_k_contributors: int = ...,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...
    def relocate_gaussians(
        self,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        ratios: torch.Tensor,
        binomial_coeffs: torch.Tensor,
        n_max: int,
        min_opacity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def add_noise_to_means(self, noise_scale: float, t: float = ..., k: float = ...) -> None: ...
    def reset_accumulated_gradient_state(self) -> None: ...
    def save_ply(self, filename: str, metadata: dict[str, str | int | float | torch.Tensor] | None) -> None: ...
    @staticmethod
    def from_ply(
        filename: str, device: torch.device = ...
    ) -> tuple[GaussianSplat3d, dict[str, str | int | float | torch.Tensor]]: ...
    def set_state(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
    ) -> None: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...
    @property
    def accumulated_gradient_step_counts(self) -> torch.Tensor: ...
    @property
    def accumulated_max_2d_radii(self) -> torch.Tensor: ...
    @property
    def accumulated_mean_2d_gradient_norms(self) -> torch.Tensor: ...
    @property
    def num_channels(self) -> int: ...
    @property
    def num_gaussians(self) -> int: ...
    @property
    def num_sh_bases(self) -> int: ...
    @property
    def opacities(self) -> torch.Tensor: ...
    @property
    def scales(self) -> torch.Tensor: ...

class GridBatchData:
    MAX_GRIDS_PER_BATCH: ClassVar[int] = ...

    # Scalar properties
    @property
    def grid_count(self) -> int: ...
    @property
    def total_voxels(self) -> int: ...
    @property
    def total_leaves(self) -> int: ...
    @property
    def total_bytes(self) -> int: ...
    @property
    def max_voxels_per_grid(self) -> int: ...
    @property
    def max_leaves_per_grid(self) -> int: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def is_empty(self) -> bool: ...
    @property
    def is_contiguous(self) -> bool: ...

    # Tensor properties
    @property
    def joffsets(self) -> torch.Tensor: ...
    @property
    def jlidx(self) -> torch.Tensor: ...
    @property
    def jidx(self) -> torch.Tensor: ...
    @property
    def num_voxels(self) -> torch.Tensor: ...
    @property
    def cum_voxels(self) -> torch.Tensor: ...
    @property
    def num_bytes(self) -> torch.Tensor: ...
    @property
    def num_leaves(self) -> torch.Tensor: ...
    @property
    def voxel_sizes(self) -> torch.Tensor: ...
    @property
    def origins(self) -> torch.Tensor: ...
    @property
    def bbox(self) -> torch.Tensor: ...
    @property
    def dual_bbox(self) -> torch.Tensor: ...
    @property
    def total_bbox(self) -> torch.Tensor: ...
    @property
    def voxel_to_world_matrices(self) -> torch.Tensor: ...
    @property
    def world_to_voxel_matrices(self) -> torch.Tensor: ...

    # Per-grid queries
    def num_voxels_at(self, bi: int) -> int: ...
    def cum_voxels_at(self, bi: int) -> int: ...
    def num_bytes_at(self, bi: int) -> int: ...
    def num_leaves_at(self, bi: int) -> int: ...
    def voxel_size_at(self, bi: int) -> torch.Tensor: ...
    def origin_at(self, bi: int) -> torch.Tensor: ...
    def bbox_at(self, bi: int) -> torch.Tensor: ...
    def dual_bbox_at(self, bi: int) -> torch.Tensor: ...
    def voxel_to_world_matrix_at(self, bi: int) -> torch.Tensor: ...
    def world_to_voxel_matrix_at(self, bi: int) -> torch.Tensor: ...

    # Utility
    def jagged_like(self, data: torch.Tensor) -> JaggedTensor: ...
    def is_same(self, other: "GridBatchData") -> bool: ...

# ---------------------------------------------------------------------------
# Grid construction (module-level)
# ---------------------------------------------------------------------------

def create_from_points(
    points: JaggedTensor,
    voxel_sizes: list[list[float]],
    origins: list[list[float]],
) -> GridBatchData: ...
def create_from_ijk(
    ijk: JaggedTensor,
    voxel_sizes: list[list[float]],
    origins: list[list[float]],
) -> GridBatchData: ...
def create_from_mesh(
    vertices: JaggedTensor,
    faces: JaggedTensor,
    voxel_sizes: list[list[float]],
    origins: list[list[float]],
) -> GridBatchData: ...
def create_from_nearest_voxels_to_points(
    points: JaggedTensor,
    voxel_sizes: list[list[float]],
    origins: list[list[float]],
) -> GridBatchData: ...
def create_from_empty(
    device: torch.device,
    voxel_size: list[float],
    origin: list[float],
) -> GridBatchData: ...
def create_dense(
    num_grids: int,
    device: torch.device,
    dense_dims: list[int],
    ijk_min: list[int],
    voxel_sizes: list[list[float]],
    origins: list[list[float]],
    mask: torch.Tensor | None = ...,
) -> GridBatchData: ...
def deserialize_grid(serialized: torch.Tensor) -> GridBatchData: ...
def make_contiguous(input: GridBatchData) -> GridBatchData: ...
def concatenate_grids(elements: list[GridBatchData]) -> GridBatchData: ...

# ---------------------------------------------------------------------------
# Grid ops (module-level)
# ---------------------------------------------------------------------------

# Interpolation: forward
def sample_trilinear(
    grid: GridBatchData,
    points: JaggedTensor,
    data: torch.Tensor,
) -> list[torch.Tensor]: ...
def sample_bezier(
    grid: GridBatchData,
    points: JaggedTensor,
    data: torch.Tensor,
) -> list[torch.Tensor]: ...
def splat_trilinear(
    grid: GridBatchData,
    points: JaggedTensor,
    data: torch.Tensor,
) -> torch.Tensor: ...
def splat_bezier(
    grid: GridBatchData,
    points: JaggedTensor,
    data: torch.Tensor,
) -> torch.Tensor: ...

# Interpolation: with-gradient forward
def sample_trilinear_with_grad(
    grid: GridBatchData,
    points: JaggedTensor,
    data: torch.Tensor,
) -> list[torch.Tensor]: ...
def sample_bezier_with_grad(
    grid: GridBatchData,
    points: JaggedTensor,
    data: torch.Tensor,
) -> list[torch.Tensor]: ...

# Interpolation: with-gradient backward
def sample_trilinear_with_grad_bwd(
    grid: GridBatchData,
    points: JaggedTensor,
    data: torch.Tensor,
    grad_out_features: torch.Tensor,
    grad_out_grad_features: torch.Tensor,
) -> torch.Tensor: ...
def sample_bezier_with_grad_bwd(
    grid: GridBatchData,
    points: JaggedTensor,
    grad_out_features: torch.Tensor,
    grad_out_grad_features: torch.Tensor,
    data: torch.Tensor,
) -> torch.Tensor: ...

# Transforms
def voxel_to_world(
    grid: GridBatchData,
    points: JaggedTensor,
    is_primal: bool,
) -> torch.Tensor: ...
def world_to_voxel(
    grid: GridBatchData,
    points: JaggedTensor,
    is_primal: bool,
) -> torch.Tensor: ...
def voxel_to_world_bwd(
    grid: GridBatchData,
    grad_out: JaggedTensor,
    is_primal: bool,
) -> torch.Tensor: ...
def world_to_voxel_bwd(
    grid: GridBatchData,
    grad_out: JaggedTensor,
    is_primal: bool,
) -> torch.Tensor: ...

# Pooling
def max_pool(
    fine_grid: GridBatchData,
    coarse_grid: GridBatchData,
    data: torch.Tensor,
    factor: list[int],
    stride: list[int],
) -> torch.Tensor: ...
def max_pool_bwd(
    coarse_grid: GridBatchData,
    fine_grid: GridBatchData,
    fine_data: torch.Tensor,
    coarse_grad_out: torch.Tensor,
    factor: list[int],
    stride: list[int],
) -> torch.Tensor: ...
def avg_pool(
    fine_grid: GridBatchData,
    coarse_grid: GridBatchData,
    data: torch.Tensor,
    factor: list[int],
    stride: list[int],
) -> torch.Tensor: ...
def avg_pool_bwd(
    coarse_grid: GridBatchData,
    fine_grid: GridBatchData,
    fine_data: torch.Tensor,
    coarse_grad_out: torch.Tensor,
    factor: list[int],
    stride: list[int],
) -> torch.Tensor: ...
def refine(
    coarse_grid: GridBatchData,
    fine_grid: GridBatchData,
    data: torch.Tensor,
    factor: list[int],
) -> torch.Tensor: ...
def refine_bwd(
    fine_grid: GridBatchData,
    coarse_grid: GridBatchData,
    grad_out: torch.Tensor,
    coarse_data: torch.Tensor,
    factor: list[int],
) -> torch.Tensor: ...

# Dense I/O
def inject_op(
    dst_grid: GridBatchData,
    src_grid: GridBatchData,
    dst: JaggedTensor,
    src: JaggedTensor,
) -> None: ...
def inject_from_dense_cminor(
    grid: GridBatchData,
    dense_data: torch.Tensor,
    origins: torch.Tensor,
) -> torch.Tensor: ...
def inject_from_dense_cmajor(
    grid: GridBatchData,
    dense_data: torch.Tensor,
    origins: torch.Tensor,
) -> torch.Tensor: ...
def inject_to_dense_cminor(
    grid: GridBatchData,
    sparse_data: torch.Tensor,
    origins: torch.Tensor,
    grid_size: list[int],
) -> torch.Tensor: ...
def inject_to_dense_cmajor(
    grid: GridBatchData,
    sparse_data: torch.Tensor,
    origins: torch.Tensor,
    grid_size: list[int],
) -> torch.Tensor: ...

# Spatial queries
def points_in_grid(grid: GridBatchData, points: JaggedTensor) -> JaggedTensor: ...
def coords_in_grid(grid: GridBatchData, coords: JaggedTensor) -> JaggedTensor: ...
def cubes_in_grid(
    grid: GridBatchData,
    cube_centers: JaggedTensor,
    pad_min: list[float],
    pad_max: list[float],
) -> JaggedTensor: ...
def cubes_intersect_grid(
    grid: GridBatchData,
    cube_centers: JaggedTensor,
    pad_min: list[float],
    pad_max: list[float],
) -> JaggedTensor: ...
def ijk_to_index(grid: GridBatchData, ijk: JaggedTensor, cumulative: bool) -> JaggedTensor: ...
def ijk_to_inv_index(grid: GridBatchData, ijk: JaggedTensor, cumulative: bool) -> JaggedTensor: ...
def neighbor_indexes(
    grid: GridBatchData,
    coords: JaggedTensor,
    extent: int,
    shift: int,
) -> JaggedTensor: ...
def active_grid_coords(grid: GridBatchData) -> JaggedTensor: ...

# Ray ops
def voxels_along_rays(
    grid: GridBatchData,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_voxels: int,
    eps: float,
    return_ijk: bool,
    cumulative: bool,
) -> list[JaggedTensor]: ...
def segments_along_rays(
    grid: GridBatchData,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_segments: int,
    eps: float,
) -> JaggedTensor: ...
def uniform_ray_samples(
    grid: GridBatchData,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    t_min: JaggedTensor,
    t_max: JaggedTensor,
    min_step_size: float,
    cone_angle: float,
    include_end_segments: bool,
    return_midpoint: bool,
    eps: float,
) -> JaggedTensor: ...
def ray_implicit_intersection(
    grid: GridBatchData,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    grid_scalars: JaggedTensor,
    eps: float,
) -> JaggedTensor: ...

# Meshing / TSDF
def marching_cubes(
    grid: GridBatchData,
    field: JaggedTensor,
    level: float,
) -> list[JaggedTensor]: ...
def integrate_tsdf(
    grid: GridBatchData,
    truncation_margin: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: JaggedTensor,
    weights: JaggedTensor,
    depth_images: torch.Tensor,
    weight_images: torch.Tensor | None,
) -> tuple[GridBatchData, JaggedTensor, JaggedTensor]: ...
def integrate_tsdf_with_features(
    grid: GridBatchData,
    truncation_margin: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: JaggedTensor,
    features: JaggedTensor,
    weights: JaggedTensor,
    depth_images: torch.Tensor,
    feature_images: torch.Tensor,
    weight_images: torch.Tensor | None,
) -> tuple[GridBatchData, JaggedTensor, JaggedTensor, JaggedTensor]: ...

# Topology / misc
def grid_edge_network(
    grid: GridBatchData,
    return_voxel_coordinates: bool,
) -> list[JaggedTensor]: ...
def serialize_encode(
    grid: GridBatchData,
    order: str,
    offset: list[int],
) -> JaggedTensor: ...

# Topology ops
def coarsen_grid(grid: GridBatchData, coarsening_factor: list[int]) -> GridBatchData: ...
def upsample_grid(
    grid: GridBatchData,
    upsample_factor: list[int],
    mask: JaggedTensor | None = ...,
) -> GridBatchData: ...
def dual_grid(grid: GridBatchData, exclude_border: bool = ...) -> GridBatchData: ...
def clip_grid(
    grid: GridBatchData,
    ijk_min: list[list[int]],
    ijk_max: list[list[int]],
) -> GridBatchData: ...
def clip_grid_with_mask(
    grid: GridBatchData,
    ijk_min: list[list[int]],
    ijk_max: list[list[int]],
) -> tuple[GridBatchData, JaggedTensor]: ...
def clip_grid_features_with_mask(
    grid: GridBatchData,
    features: JaggedTensor,
    ijk_min: list[list[int]],
    ijk_max: list[list[int]],
) -> tuple[JaggedTensor, GridBatchData]: ...
def dilate_grid(grid: GridBatchData, dilation: list[int]) -> GridBatchData: ...
def merge_grids(grid1: GridBatchData, grid2: GridBatchData) -> GridBatchData: ...
def prune_grid(grid: GridBatchData, mask: JaggedTensor) -> GridBatchData: ...
def conv_grid(
    grid: GridBatchData,
    kernel_size: list[int],
    stride: list[int],
) -> GridBatchData: ...
def conv_transpose_grid(
    grid: GridBatchData,
    kernel_size: list[int],
    stride: list[int],
) -> GridBatchData: ...

# Batch ops
def clone_grid(grid: GridBatchData, device: torch.device) -> GridBatchData: ...
def serialize_grid(grid: GridBatchData) -> torch.Tensor: ...
def index_grid_int(grid: GridBatchData, index: int) -> GridBatchData: ...
def index_grid_slice(
    grid: GridBatchData,
    start: int,
    stop: int,
    step: int,
) -> GridBatchData: ...
def index_grid_tensor(grid: GridBatchData, indices: torch.Tensor) -> GridBatchData: ...
def index_grid_int64_list(grid: GridBatchData, indices: list[int]) -> GridBatchData: ...
def index_grid_bool_list(grid: GridBatchData, indices: list[bool]) -> GridBatchData: ...

class JaggedTensor:
    jdata: torch.Tensor
    requires_grad: bool
    @overload
    def __init__(self, tensor_list: list[list[torch.Tensor]]) -> None: ...
    @overload
    def __init__(self, tensor_list: list[torch.Tensor]) -> None: ...
    @overload
    def __init__(self, tensor: torch.Tensor) -> None: ...
    def abs(self) -> JaggedTensor: ...
    def abs_(self) -> JaggedTensor: ...
    def ceil(self) -> JaggedTensor: ...
    def ceil_(self) -> JaggedTensor: ...
    def clone(self) -> JaggedTensor: ...
    def cpu(self) -> JaggedTensor: ...
    def cuda(self) -> JaggedTensor: ...
    def detach(self) -> JaggedTensor: ...
    def double(self) -> JaggedTensor: ...
    def float(self) -> JaggedTensor: ...
    def floor(self) -> JaggedTensor: ...
    def floor_(self) -> JaggedTensor: ...
    @staticmethod
    def from_data_and_indices(arg0: torch.Tensor, arg1: torch.Tensor, arg2: int) -> JaggedTensor: ...
    @staticmethod
    def from_data_and_offsets(arg0: torch.Tensor, arg1: torch.Tensor) -> JaggedTensor: ...
    @staticmethod
    def from_data_indices_and_list_ids(
        data: torch.Tensor, indices: torch.Tensor, list_ids: torch.Tensor, num_tensors: int
    ) -> JaggedTensor: ...
    @staticmethod
    def from_data_offsets_and_list_ids(
        data: torch.Tensor, offsets: torch.Tensor, list_ids: torch.Tensor
    ) -> JaggedTensor: ...
    def int(self) -> JaggedTensor: ...
    def jagged_like(self, data: torch.Tensor) -> JaggedTensor: ...
    def jflatten(self, dim: int = ...) -> JaggedTensor: ...
    def jmax(self, dim: int = ..., keepdim: bool = ...) -> list[JaggedTensor]: ...
    def jmin(self, dim: int = ..., keepdim: bool = ...) -> list[JaggedTensor]: ...
    @overload
    def jreshape(self, lshape: list[int]) -> JaggedTensor: ...
    @overload
    def jreshape(self, lshape: list[list[int]]) -> JaggedTensor: ...
    def jreshape_as(self, other: JaggedTensor | torch.Tensor) -> JaggedTensor: ...
    def jsqueeze(self, dim: int | None = None) -> JaggedTensor: ...
    def jsum(self, dim: int = ..., keepdim: bool = ...) -> JaggedTensor: ...
    def long(self) -> JaggedTensor: ...
    def requires_grad_(self, arg0: bool) -> JaggedTensor: ...
    def rmask(self, mask: torch.Tensor) -> JaggedTensor: ...
    def round(self, decimals: int = ...) -> JaggedTensor: ...
    def round_(self, decimals: int = ...) -> JaggedTensor: ...
    def sqrt(self) -> JaggedTensor: ...
    def sqrt_(self) -> JaggedTensor: ...
    @overload
    def to(self, arg0: torch.device) -> JaggedTensor: ...
    @overload
    def to(self, arg0: str) -> JaggedTensor: ...
    @overload
    def to(self, arg0: torch.dtype) -> JaggedTensor: ...
    @overload
    def to(self, device: torch.device) -> JaggedTensor: ...
    @overload
    def to(self, device: str) -> JaggedTensor: ...
    def type(self, arg0: torch.dtype) -> JaggedTensor: ...
    def type_as(self, arg0: JaggedTensor | torch.Tensor) -> JaggedTensor: ...
    def unbind(self) -> ListOfTensors | ListOfListsOfTensors: ...
    @overload
    def __add__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __add__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __add__(self, other: int) -> JaggedTensor: ...
    @overload
    def __add__(self, other: float) -> JaggedTensor: ...
    @overload
    def __eq__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __eq__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __eq__(self, other: int) -> JaggedTensor: ...
    @overload
    def __eq__(self, other: float) -> JaggedTensor: ...
    @overload
    def __floordiv__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __floordiv__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __floordiv__(self, other: int) -> JaggedTensor: ...
    @overload
    def __floordiv__(self, other: float) -> JaggedTensor: ...
    @overload
    def __ge__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __ge__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __ge__(self, other: int) -> JaggedTensor: ...
    @overload
    def __ge__(self, other: float) -> JaggedTensor: ...
    def __getitem__(self, arg0) -> JaggedTensor: ...
    @overload
    def __gt__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __gt__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __gt__(self, other: int) -> JaggedTensor: ...
    @overload
    def __gt__(self, other: float) -> JaggedTensor: ...
    @overload
    def __iadd__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __iadd__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __iadd__(self, other: int) -> JaggedTensor: ...
    @overload
    def __iadd__(self, other: float) -> JaggedTensor: ...
    @overload
    def __ifloordiv__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __ifloordiv__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __ifloordiv__(self, other: int) -> JaggedTensor: ...
    @overload
    def __ifloordiv__(self, other: float) -> JaggedTensor: ...
    @overload
    def __imod__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __imod__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __imod__(self, other: int) -> JaggedTensor: ...
    @overload
    def __imod__(self, other: float) -> JaggedTensor: ...
    @overload
    def __imul__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __imul__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __imul__(self, other: int) -> JaggedTensor: ...
    @overload
    def __imul__(self, other: float) -> JaggedTensor: ...
    @overload
    def __ipow__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __ipow__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __ipow__(self, other: int) -> JaggedTensor: ...
    @overload
    def __ipow__(self, other: float) -> JaggedTensor: ...
    @overload
    def __isub__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __isub__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __isub__(self, other: int) -> JaggedTensor: ...
    @overload
    def __isub__(self, other: float) -> JaggedTensor: ...
    @overload
    def __itruediv__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __itruediv__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __itruediv__(self, other: int) -> JaggedTensor: ...
    @overload
    def __itruediv__(self, other: float) -> JaggedTensor: ...
    @overload
    def __le__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __le__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __le__(self, other: int) -> JaggedTensor: ...
    @overload
    def __le__(self, other: float) -> JaggedTensor: ...
    def __len__(self) -> int: ...
    @overload
    def __lt__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __lt__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __lt__(self, other: int) -> JaggedTensor: ...
    @overload
    def __lt__(self, other: float) -> JaggedTensor: ...
    @overload
    def __mod__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __mod__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __mod__(self, other: int) -> JaggedTensor: ...
    @overload
    def __mod__(self, other: float) -> JaggedTensor: ...
    @overload
    def __mul__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __mul__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __mul__(self, other: int) -> JaggedTensor: ...
    @overload
    def __mul__(self, other: float) -> JaggedTensor: ...
    @overload
    def __ne__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __ne__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __ne__(self, other: int) -> JaggedTensor: ...
    @overload
    def __ne__(self, other: float) -> JaggedTensor: ...
    @overload
    def __neg__(self) -> JaggedTensor: ...
    @overload
    def __neg__(self) -> JaggedTensor: ...
    @overload
    def __pow__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __pow__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __pow__(self, other: int) -> JaggedTensor: ...
    @overload
    def __pow__(self, other: float) -> JaggedTensor: ...
    @overload
    def __sub__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __sub__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __sub__(self, other: int) -> JaggedTensor: ...
    @overload
    def __sub__(self, other: float) -> JaggedTensor: ...
    @overload
    def __truediv__(self, other: torch.Tensor) -> JaggedTensor: ...
    @overload
    def __truediv__(self, other: JaggedTensor) -> JaggedTensor: ...
    @overload
    def __truediv__(self, other: int) -> JaggedTensor: ...
    @overload
    def __truediv__(self, other: float) -> JaggedTensor: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def edim(self) -> int: ...
    @property
    def eshape(self) -> list[int]: ...
    @property
    def is_cpu(self) -> bool: ...
    @property
    def is_cuda(self) -> bool: ...
    @property
    def jidx(self) -> torch.Tensor: ...
    @property
    def jlidx(self) -> torch.Tensor: ...
    @property
    def joffsets(self) -> torch.Tensor: ...
    @property
    def ldim(self) -> int: ...
    @property
    def lshape(self) -> list[int] | list[list[int]]: ...
    @property
    def num_tensors(self) -> int: ...
    @property
    def rshape(self) -> tuple[int, ...]: ...
    def __iter__(self) -> typing.Iterator[JaggedTensor]: ...

class ProjectedGaussianSplats:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def antialias(self) -> bool: ...
    @property
    def conics(self) -> torch.Tensor: ...
    @property
    def depths(self) -> torch.Tensor: ...
    @property
    def eps_2d(self) -> float: ...
    @property
    def far_plane(self) -> float: ...
    @property
    def image_height(self) -> int: ...
    @property
    def image_width(self) -> int: ...
    @property
    def means2d(self) -> torch.Tensor: ...
    @property
    def min_radius_2d(self) -> float: ...
    @property
    def near_plane(self) -> float: ...
    @property
    def opacities(self) -> torch.Tensor: ...
    @property
    def projection_type(self) -> GaussianSplat3d.ProjectionType: ...
    @property
    def radii(self) -> torch.Tensor: ...
    @property
    def render_quantities(self) -> torch.Tensor: ...
    @property
    def sh_degree_to_use(self) -> int: ...
    @property
    def tile_gaussian_ids(self) -> torch.Tensor: ...
    @property
    def tile_offsets(self) -> torch.Tensor: ...

class GaussianSplat3dView:
    @property
    def tile_size(self) -> int: ...
    @tile_size.setter
    def tile_size(self, value: int) -> None: ...
    @property
    def min_radius_2d(self) -> float: ...
    @min_radius_2d.setter
    def min_radius_2d(self, value: float) -> None: ...
    @property
    def eps_2d(self) -> float: ...
    @eps_2d.setter
    def eps_2d(self, value: float) -> None: ...
    @property
    def antialias(self) -> bool: ...
    @antialias.setter
    def antialias(self, value: bool) -> None: ...
    @property
    def rgb_rgb_rgb_sh(self) -> bool: ...
    @rgb_rgb_rgb_sh.setter
    def rgb_rgb_rgb_sh(self, value: bool) -> None: ...
    @property
    def sh_degree_to_use(self) -> int: ...
    @sh_degree_to_use.setter
    def sh_degree_to_use(self, value: int) -> None: ...
    @property
    def sh_stride_rrr_ggg_bbb(self) -> bool: ...
    @sh_stride_rrr_ggg_bbb.setter
    def sh_stride_rrr_ggg_bbb(self, value: bool) -> None: ...

class CameraView:
    @property
    def visible(self) -> bool: ...
    @visible.setter
    def visible(self, value: bool) -> None: ...
    @property
    def axis_length(self) -> float: ...
    @axis_length.setter
    def axis_length(self, value: float) -> None: ...
    @property
    def axis_thickness(self) -> float: ...
    @axis_thickness.setter
    def axis_thickness(self, value: float) -> None: ...
    @property
    def frustum_line_width(self) -> float: ...
    @frustum_line_width.setter
    def frustum_line_width(self, value: float) -> None: ...
    @property
    def frustum_scale(self) -> float: ...
    @frustum_scale.setter
    def frustum_scale(self, value: float) -> None: ...
    @property
    def frustum_color(self) -> tuple[float, float, float]: ...
    @frustum_color.setter
    def frustum_color(self, value: tuple[float, float, float]) -> None: ...

class Viewer:
    def __init__(self, ip_address: str, port: int, device_id: int, verbose: bool) -> None: ...
    def port(self) -> int: ...
    def ip_address(self) -> str: ...
    def reset(self) -> None: ...
    def add_scene(self, scene_name: str) -> None: ...
    def remove_scene(self, scene_name: str) -> None: ...
    def remove_view(self, scene_name: str, name: str) -> None: ...
    def add_gaussian_splat_3d_view(
        self, scene_name: str, name: str, gaussian_splat_3d: GaussianSplat3d
    ) -> GaussianSplat3dView: ...
    def has_gaussian_splat_3d_view(self, name: str) -> bool: ...
    def get_gaussian_splat_3d_view(self, name: str) -> GaussianSplat3dView: ...
    def camera_orbit_center(self, scene_name: str) -> tuple[float, float, float]: ...
    def set_camera_orbit_center(self, scene_name: str, ox: float, oy: float, oz: float) -> None: ...
    def camera_orbit_radius(self, scene_name: str) -> float: ...
    def set_camera_orbit_radius(self, scene_name: str, radius: float) -> None: ...
    def camera_view_direction(self, scene_name: str) -> tuple[float, float, float]: ...
    def set_camera_view_direction(self, scene_name: str, dx: float, dy: float, dz: float) -> None: ...
    def camera_up_direction(self, scene_name: str) -> tuple[float, float, float]: ...
    def set_camera_up_direction(self, scene_name: str, ux: float, uy: float, uz: float) -> None: ...
    def camera_near(self, scene_name: str) -> float: ...
    def set_camera_near(self, scene_name: str, near: float) -> None: ...
    def camera_far(self, scene_name: str) -> float: ...
    def set_camera_far(self, scene_name: str, far: float) -> None: ...
    def camera_projection_type(self, scene_name: str) -> str: ...
    def set_camera_projection_type(self, scene_name: str, projection_type: GaussianSplat3d.ProjectionType) -> None: ...
    def add_camera_view(
        self,
        scene_name: str,
        name: str,
        camera_to_world_matrices: NumericMaxRank3,
        projection_matrices: NumericMaxRank3,
        image_sizes: NumericMaxRank2,
        frustum_near_plane: float,
        frustum_far_plane: float,
        axis_length: float,
        axis_thickness: float,
        frustum_line_width: float,
        frustum_scale: float,
        frustum_color: tuple[float, float, float],
        visible: bool,
    ) -> CameraView: ...
    def has_camera_view(self, name: str) -> bool: ...
    def get_camera_view(self, name: str) -> CameraView: ...
    def add_image(
        self,
        scene_name: str,
        name: str,
        rgba_image: NumericMaxRank1,
        width: int,
        height: int,
    ) -> None: ...
    def wait_for_interrupt(self) -> None: ...

class config:
    enable_ultra_sparse_acceleration: ClassVar[bool] = ...
    pedantic_error_checking: ClassVar[bool] = ...
    def __init__(self, *args, **kwargs) -> None: ...

def gaussian_render_jagged(
    means: JaggedTensor,
    quats: JaggedTensor,
    scales: JaggedTensor,
    opacities: JaggedTensor,
    sh_coeffs: JaggedTensor,
    viewmats: JaggedTensor,
    Ks: JaggedTensor,
    image_width: int,
    image_height: int,
    near_plane: float = ...,
    far_plane: float = ...,
    sh_degree_to_use: int = ...,
    tile_size: int = ...,
    radius_clip: float = ...,
    eps2d: float = ...,
    antialias: bool = ...,
    render_depth_channel: bool = ...,
    return_debug_info: bool = ...,
    render_depth_only: bool = ...,
    ortho: bool = ...,
    backgrounds: Optional[torch.Tensor] = ...,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: ...
def evaluate_spherical_harmonics(
    sh_degree: int,
    num_cameras: int,
    sh0: torch.Tensor,
    radii: torch.Tensor,
    shN: Optional[torch.Tensor] = ...,
    view_directions: Optional[torch.Tensor] = ...,
) -> torch.Tensor: ...
@overload
def jcat(grid_batches: list[GridBatchData]) -> GridBatchData: ...
@overload
def jcat(jagged_tensors: list[JaggedTensor | torch.Tensor], dim: int | None = ...) -> JaggedTensor: ...
def jempty(
    lshape: LShapeSpec,
    rshape: RShapeSpec | None = ...,
    dtype: torch.dtype | None = ...,
    device: DeviceIdentifier | None = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> JaggedTensor: ...
def jones(
    lshape: LShapeSpec,
    rshape: RShapeSpec | None = ...,
    dtype: torch.dtype | None = ...,
    device: DeviceIdentifier | None = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> JaggedTensor: ...
def jrand(
    lshape: LShapeSpec,
    rshape: RShapeSpec | None = ...,
    dtype: torch.dtype | None = ...,
    device: DeviceIdentifier | None = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> JaggedTensor: ...
def jrandn(
    lshape: LShapeSpec,
    rshape: RShapeSpec | None = ...,
    dtype: torch.dtype | None = ...,
    device: DeviceIdentifier | None = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> JaggedTensor: ...
def jzeros(
    lshape: LShapeSpec,
    rshape: RShapeSpec | None = ...,
    dtype: torch.dtype | None = ...,
    device: DeviceIdentifier | None = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> JaggedTensor: ...
@overload
def load(
    path: str,
    indices: list[int],
    device: torch.device = ...,
    verbose: bool = ...,
) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(
    path: str,
    indices: list[int],
    device: str = ...,
    verbose: bool = ...,
) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(
    path: str,
    index: int,
    device: torch.device = ...,
    verbose: bool = ...,
) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(
    path: str,
    index: int,
    device: str = ...,
    verbose: bool = ...,
) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(
    path: str,
    names: list[str],
    device: torch.device = ...,
    verbose: bool = ...,
) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(
    path: str,
    names: list[str],
    device: str = ...,
    verbose: bool = ...,
) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(
    path: str,
    name: str,
    device: torch.device = ...,
    verbose: bool = ...,
) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(
    path: str,
    name: str,
    device: str = ...,
    verbose: bool = ...,
) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(path: str, device: torch.device = ..., verbose: bool = ...) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def load(path: str, device: str = ..., verbose: bool = ...) -> tuple[GridBatchData, JaggedTensor, list[str]]: ...
@overload
def save(
    path: str,
    grid_batch: GridBatchData,
    data: JaggedTensor | None = ...,
    names: list[str] = ...,
    compressed: bool = ...,
    verbose: bool = ...,
) -> None: ...
@overload
def save(
    path: str,
    grid_batch: GridBatchData,
    data: JaggedTensor | None = ...,
    name: str = ...,
    compressed: bool = ...,
    verbose: bool = ...,
) -> None: ...
def morton(ijk: torch.Tensor) -> torch.Tensor: ...
def hilbert(ijk: torch.Tensor) -> torch.Tensor: ...
def scaled_dot_product_attention(
    query: JaggedTensor | torch.Tensor,
    key: JaggedTensor | torch.Tensor,
    value: JaggedTensor | torch.Tensor,
    scale: float,
) -> JaggedTensor: ...
def volume_render(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    deltaTs: torch.Tensor,
    ts: torch.Tensor,
    packInfo: torch.Tensor,
    transmittanceThresh: float,
) -> list[torch.Tensor]: ...

class RollingShutterType(Enum):
    NONE = ...
    VERTICAL = ...
    HORIZONTAL = ...

class DistortionModel(Enum):
    PINHOLE = ...
    OPENCV_RADTAN_5 = ...
    OPENCV_RATIONAL_8 = ...
    OPENCV_RADTAN_THIN_PRISM_9 = ...
    OPENCV_THIN_PRISM_12 = ...
    ORTHOGRAPHIC = ...
